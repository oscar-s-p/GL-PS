"""
Ptyhon module with function definitions for flux ratio modeling
"""

import pickle
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns     # type: ignore
import tensorflow as tf   # type: ignore
from corner import corner
import tensorflow_probability as tfp
from tensorflow_probability import (
    distributions as tfd,
    bijectors as tfb,
    experimental as tfe,
)
tfd = tfp.distributions

import tqdm
from tqdm.auto import trange

from lenstronomy.Data.pixel_grid import PixelGrid


import gigalens.profile
from gigalens.tf.inference import ModellingSequence
from gigalens.tf.model import ForwardProbModel, BackwardProbModel
from gigalens.model import PhysicalModel
from gigalens.tf.simulator import LensSimulator
from gigalens.simulator import SimulatorConfig
from gigalens.tf.profiles.light import sersic
from gigalens.tf.profiles.mass import sis, shear, epl, sie
import multiprocessing
import time

"""
Delensing functions
"""

@tf.function
def _rotate(x, y, phi):
    cos_phi, sin_phi = tf.cos(phi), tf.sin(phi)
    return x * cos_phi + y * sin_phi, -x * sin_phi + y * cos_phi

@tf.function
def deriv_epl(x, y, theta_E, gamma, e1, e2, center_x, center_y):
  phi = tf.atan2(e2, e1) / 2
  c = tf.clip_by_value(tf.math.sqrt(e1 ** 2 + e2 ** 2), 0, 1)
  q = (1 - c) / (1 + c) # type: ignore
  theta_E_conv = theta_E / (tf.math.sqrt((1.0 + q ** 2) / (2.0 * q)))
  b = theta_E_conv * tf.math.sqrt((1 + q ** 2) / 2)
  t = gamma - 1

  x, y = x - center_x, y - center_y
  x, y = _rotate(x, y, phi) # type: ignore


  R = tf.clip_by_value(tf.math.sqrt((q * x) ** 2 + y ** 2), 1e-10, 1e10)
  angle = tf.math.atan2(y, q * x)
  f = (1 - q) / (1 + q)
  Cs, Ss = tf.math.cos(angle), tf.math.sin(angle)
  Cs2, Ss2 = tf.math.cos(2 * angle), tf.math.sin(2 * angle)

  niter = tf.stop_gradient(tf.math.log(1e-12) / tf.math.log(tf.reduce_max(f)) + 2)

  def body(n, p):
      last_x, last_y, f_x, f_y = p
      prefac = -f * (2 * n - (2 - t)) / (2 * n + (2 - t))
      last_x, last_y = prefac * (Cs2 * last_x - Ss2 * last_y), prefac * (
              Ss2 * last_x + Cs2 * last_y
      )
      return n + 1, (last_x, last_y, f_x + last_x, f_y + last_y)

  _, _, fx, fy = tf.while_loop(
      lambda i, p: i < niter,
      body, (1.0, (Cs, Ss, Cs, Ss)),
      maximum_iterations = 500,
      swap_memory=True,
  )[1] # type: ignore

  prefac = (2 * b) / (1 + q) * tf.math.pow(b / R, t - 1)
  fx, fy = fx * prefac, fy * prefac
  return _rotate(fx, fy, -phi)

@tf.function
def deriv_shear(x, y, gamma1, gamma2):
  return gamma1 * x + gamma2 * y, gamma2 * x - gamma1 * y

##########################################################
@tf.function
def _deriv_epl_shear(x, y, lens_params):
  f_xi1, f_yi1 = deriv_epl(x, y, **lens_params[0][0])   # type: ignore
  f_xi2, f_yi2 = deriv_shear(x, y, **lens_params[0][1]) # type: ignore
  f_xi, f_yi = f_xi1 + f_xi2, f_yi1 + f_yi2
  return f_xi,f_yi
#########################################################

@tf.function
def _beta_epl_shear(x, y, lens_params):

 f_xi, f_yi = deriv_epl(x, y, **lens_params[0][0])     # type: ignore
 beta_x, beta_y = x - f_xi, y - f_yi

 f_xi, f_yi = deriv_shear(x, y, **lens_params[0][1])   # type: ignore
 beta_x, beta_y = beta_x - f_xi, beta_y - f_yi

 return beta_x, beta_y

@tf.function
def _hessian_differential_cross(x, y, kwargs, diff=0.0001):
        """
        computes the numerical differentials over a finite range for f_xx, f_yy, f_xy from f_x and f_y
        The differentials are computed along the cross centered at (x, y).

        :param x: x-coordinate
        :param y: y-coordinate
        :param kwargs: lens model keyword argument list
        :param k: int, list of bools or None, indicating a subset of lens models to be evaluated
        :param diff: float, scale of the finite differential (diff/2 in each direction used to compute the differential
        :return: f_xx, f_xy, f_yx, f_yy
        """
        # alpha_ra_dx, alpha_dec_dx = _beta_EPL_shear(x + diff/2, y, kwargs)
        # alpha_ra_dy, alpha_dec_dy = _beta_EPL_shear(x, y + diff/2, kwargs)

        # alpha_ra_dx_, alpha_dec_dx_ = _beta_EPL_shear(x - diff/2, y, kwargs)
        # alpha_ra_dy_, alpha_dec_dy_ = _beta_EPL_shear(x, y - diff/2, kwargs)

        alpha_ra_dx, alpha_dec_dx = _deriv_epl_shear(x + diff/2, y, kwargs) # type: ignore
        alpha_ra_dy, alpha_dec_dy = _deriv_epl_shear(x, y + diff/2, kwargs) # type: ignore

        alpha_ra_dx_, alpha_dec_dx_ = _deriv_epl_shear(x - diff/2, y, kwargs) # type: ignore
        alpha_ra_dy_, alpha_dec_dy_ = _deriv_epl_shear(x, y - diff/2, kwargs) # type: ignore

        dalpha_rara = (alpha_ra_dx - alpha_ra_dx_) / diff
        dalpha_radec = (alpha_ra_dy - alpha_ra_dy_) / diff
        dalpha_decra = (alpha_dec_dx - alpha_dec_dx_) / diff
        dalpha_decdec = (alpha_dec_dy - alpha_dec_dy_) / diff

        f_xx = dalpha_rara
        f_yy = dalpha_decdec
        f_xy = dalpha_radec
        f_yx = dalpha_decra
        return f_xx, f_xy, f_yx, f_yy


"""
Magnification functions
"""


@tf.function
def _hessian_differential_square(x, y, kwargs, diff=0.00001):
        """
        computes the numerical differentials over a finite range for f_xx, f_yy, f_xy from f_x and f_y
        The differentials are computed on the square around (x, y). This minimizes curl.

        :param x: x-coordinate
        :param y: y-coordinate
        :param kwargs: lens model keyword argument list
        :param k: int, list of booleans or None, indicating a subset of lens models to be evaluated
        :param diff: float, scale of the finite differential (diff/2 in each direction used to compute the differential
        :return: f_xx, f_xy, f_yx, f_yy
        # """
        # alpha_ra_pp, alpha_dec_pp = _beta_EPL_shear(x + diff/2, y + diff/2, kwargs)
        # alpha_ra_pn, alpha_dec_pn = _beta_EPL_shear(x + diff/2, y - diff/2, kwargs)

        # alpha_ra_np, alpha_dec_np = _beta_EPL_shear(x - diff / 2, y + diff / 2, kwargs)
        # alpha_ra_nn, alpha_dec_nn = _beta_EPL_shear(x - diff / 2, y - diff / 2, kwargs)

        alpha_ra_pp, alpha_dec_pp = _deriv_epl_shear(x + diff/2, y + diff/2, kwargs) # type: ignore
        alpha_ra_pn, alpha_dec_pn = _deriv_epl_shear(x + diff/2, y - diff/2, kwargs) # type: ignore

        alpha_ra_np, alpha_dec_np = _deriv_epl_shear(x - diff / 2, y + diff / 2, kwargs) # type: ignore
        alpha_ra_nn, alpha_dec_nn = _deriv_epl_shear(x - diff / 2, y - diff / 2, kwargs) # type: ignore

        f_xx = (alpha_ra_pp - alpha_ra_np + alpha_ra_pn - alpha_ra_nn) / diff / 2
        f_xy = (alpha_ra_pp - alpha_ra_pn + alpha_ra_np - alpha_ra_nn) / diff / 2
        f_yx = (alpha_dec_pp - alpha_dec_np + alpha_dec_pn - alpha_dec_nn) / diff / 2
        f_yy = (alpha_dec_pp - alpha_dec_pn + alpha_dec_np - alpha_dec_nn) / diff / 2

        return f_xx, f_xy, f_yx, f_yy
@tf.function
def hessian(x, y, kwargs, diff=0.00001, diff_method='square'):
        """
        hessian matrix

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :param diff: float, scale over which the finite numerical differential is computed. If None, then using the
         exact (if available) differentials.
        :param diff_method: string, 'square' or 'cross', indicating whether finite differentials are computed from a
         cross or a square of points around (x, y)
        :return: f_xx, f_xy, f_yx, f_yy components
        """
        # if diff is None:
        #     return hessian(x, y, kwargs)
        if diff_method == 'square': #elif
            return _hessian_differential_square(x, y, kwargs, diff=diff)
        elif diff_method == 'cross':
            return _hessian_differential_cross(x, y, kwargs, diff=diff)
        else:
            raise ValueError('diff_method %s not supported. Chose among "square" or "cross".' % diff_method)

@tf.function
def magnification(x, y, kwargs, diff=0.0001, diff_method='square'):
        """
        magnification
        mag = 1/det(A)
        A = 1 - d^2phi/d_ij

        :param x: image plane x-position (preferentially arcsec)
        :type x: numpy array
        :param y: image plane y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :param diff: float, scale over which the finite numerical differential is computed. If None, then using the
         exact (if available) differentials.
        :param diff_method: string, 'square' or 'cross', indicating whether finite differentials are computed from a
         cross or a square of points around (x, y)
        :return: magnification
        """

        f_xx, f_xy, f_yx, f_yy = hessian(x, y, kwargs, diff=diff, diff_method=diff_method) # type: ignore
        det_A = (1 - f_xx) * (1 - f_yy) - f_xy*f_yx
        return 1/det_A

@tf.function
def determinant(x, y, kwargs, diff=0.0001, diff_method='square'):
        """
        magnification
        mag = 1/det(A)
        A = 1 - d^2phi/d_ij

        :param x: image plane x-position (preferentially arcsec)
        :type x: numpy array
        :param y: image plane y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :param diff: float, scale over which the finite numerical differential is computed. If None, then using the
         exact (if available) differentials.
        :param diff_method: string, 'square' or 'cross', indicating whether finite differentials are computed from a
         cross or a square of points around (x, y)
        :return: magnification
        """

        f_xx, f_xy, f_yx, f_yy = hessian(x, y, kwargs, diff=diff, diff_method=diff_method) # type: ignore
        det_A = (1 - f_xx) * (1 - f_yy) - f_xy*f_yx
        return det_A

"""
Point source light class
"""
class Sersic(gigalens.profile.LightProfile):
    """A spherically symmetric Sersic light profile.

    .. math::
        I(x,y) = I_e \\exp\\left(-b_n \\left(\\left(\\frac{D(x,y)}{R_s}\\right)^{1/n} - 1\\right)\\right)

    where :math:`D(x,y)` is the distance function (as defined in :func:`~gigalens.tf.profiles.light.sersic.Sersic.distance`).
    In the simplest case, it is just Euclidean distance from the center, and when ellipticity is non-zero, the
    coordinate axes are translated, rotated and scaled to match the ellipse defined by the complex ellipticities
    ``(e1,e2)`` with center ``(center_x, center_y)`` then the Euclidean distance from the center is calculated.
    If least squares is not being used, the amplitude :math:`I_e` is set to be 1.
    """

    _name = "SERSIC"
    _params = ["R_sersic", "n_sersic", "center_x", "center_y"]

    @tf.function
    def light(self, x, y, R_sersic, n_sersic, center_x, center_y, Ie=None):
        Ie = tf.ones_like(R_sersic) if self.use_lstsq else Ie
        R = self.distance(x, y, center_x, center_y)
        bn = 1.9992 * n_sersic - 0.3271
        return Ie * tf.math.exp(-bn * ((R / R_sersic) ** (1 / n_sersic) - 1.0))


    @tf.function
    def distance(self, x, y, cx, cy, e1=None, e2=None):
        """

        Args:
            x: The :math:`x` coordinates to evaluate the distance function at
            y: The :math:`y` coordinates to evaluate the distance function at
            cx: The :math:`x` coordinate of the center of the Sersic light component
            cy: The :math:`y` coordinate of the center of the Sersic light component
            e1: Complex ellipticity component. If unspecified, it is assumed to be zero.
            e2: Complex ellipticity component. If unspecified, it is assumed to be zero.

        Returns:
            The distance function evaluated at ``(x,y)``
        """
        if e1 is None:
            e1 = tf.zeros_like(cx)
        if e2 is None:
            e2 = tf.zeros_like(cx)
        phi = tf.atan2(e2, e1) / 2
        c = tf.math.sqrt(e1 ** 2 + e2 ** 2)
        q = (1 - c) / (1 + c)
        dx, dy = x - cx, y - cy
        cos_phi, sin_phi = tf.math.cos(phi), tf.math.sin(phi)
        xt1 = (cos_phi * dx + sin_phi * dy) * tf.math.sqrt(q)
        xt2 = (-sin_phi * dx + cos_phi * dy) / tf.math.sqrt(q)
        return tf.sqrt(xt1 ** 2 + xt2 ** 2)


class SersicEllipse(Sersic):
    _name = "SERSIC_ELLIPSE"
    _params = ["R_sersic", "n_sersic", "e1", "e2", "center_x", "center_y"]

    @tf.function
    def light(self, x, y, R_sersic, n_sersic, e1, e2, center_x, center_y, Ie=None):
        Ie = tf.ones_like(R_sersic) if self.use_lstsq else Ie
        R = self.distance(x, y, center_x, center_y, e1, e2)
        bn = 1.9992 * n_sersic - 0.3271
        return Ie * tf.math.exp(-bn * ((R / R_sersic) ** (1 / n_sersic) - 1.0))

class CoreSersic(Sersic):
    _name = "CORE_SERSIC"
    _params = [
        "R_sersic",
        "n_sersic",
        "Rb",
        "alpha",
        "gamma",
        "e1",
        "e2",
        "center_x",
        "center_y",
    ]

    @tf.function
    def light(
        self,
        x,
        y,
        R_sersic,
        n_sersic,
        Rb,
        alpha,
        gamma,
        e1,
        e2,
        center_x,
        center_y,
        Ie=None,
    ):
        Ie = tf.ones_like(R_sersic) if self.use_lstsq else Ie
        R = self.distance(x, y, center_x, center_y, e1, e2)
        bn = 1.9992 * n_sersic - 0.3271
        result = (
            Ie
            * (1 + (Rb / R) ** alpha) ** (gamma / alpha)
            * tf.math.exp(
                -bn
                * (
                    (R ** alpha + Rb ** alpha)
                    / R_sersic ** alpha ** 1.0
                    / (alpha * n_sersic)
                )
                - 1.0
            )
        )
        return result


class Point(Sersic):
    _name = "POINT"
    _params = ["center_x", "center_y"]
    '''
    simulate a gaussian, return brigtest pixel
    I have not kept bn which was to ensure Ie is the high light brigtness.
    Insteady I want Ie to be the brightest flux.
    I use R_scale to basically make the light fall to zero quickly.
    I dont have a lstsq fit version of this yet.  See return statement for Sersic above.
    '''
    @tf.function
    def light(self, x, y, center_x, center_y, Ie=None, R_sersic=None, n_sersic=None):
        R = self.distance(x, y, center_x, center_y)
#         R_scale = 0.002
#         pix = Ie * jnp.exp(-(R / R_scale) ** 2 )
#         maxpix = jnp.max(pix)
#         ret = jnp.array(maxpix)
#         # return ret[jnp.newaxis, ...] if self.use_lstsq else (Ie * ret)
        n_fixed = 1000
        # R_fixed = 0.002
        R_fixed = 0.0015
        b_fixed = 1.9992 * n_fixed - 0.3271
        ret = Ie * tf.exp(-b_fixed * ((R / R_fixed) ** (1 / n_fixed) - 1.0)) # type: ignore
        return ret[tf.newaxis, ...] if self.use_lstsq else (Ie * ret)
    
"""
Bright points
"""
###### mod ified #######

class BrightestPoints:
    def __init__(self, number_of_images=4, num_pixels=750, grid_size=250, delta_pix=0.02, supersample=1):
        self.number_of_images = number_of_images
        self.num_pixels = num_pixels
        self.grid_size = grid_size
        self.delta_pix = delta_pix
        self.supersample = supersample

    def find_brightest_points(self, img):
        """Find the brightest points in a lensed point source image.
        Args:
            img (tf.Tensor): Input lensed point source image tensor.
                            Shape: (num_pixels, num_pixels).
        Returns:
            tf.Tensor: Tensor containing the coordinates of the brightest points in the image.
                      Shape: (number_of_images, 2)
        """
        image = tf.reshape(img, [1, self.num_pixels, self.num_pixels, 1])
        patches = tf.image.extract_patches(
            images=image,
            sizes=[1, self.grid_size, self.grid_size, 1],
            strides=[1, self.grid_size, self.grid_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patches = tf.reshape(
            patches,
            [int(self.num_pixels / self.grid_size), int(self.num_pixels / self.grid_size), self.grid_size, self.grid_size, 1]
        )
        max_values = tf.reduce_max(patches, axis=(2, 3))
        result_grid = tf.squeeze(max_values)
        flat_grid = tf.reshape(result_grid, [-1])
        values, indices = tf.math.top_k(flat_grid, k=self.number_of_images)

        brightest_points = []

        for i in range(self.number_of_images):

            point = tf.where(img == values[i])
            if point.shape[0] > 0:
                brightest_points.append(point[0].numpy().tolist())


        brightest_points = tf.constant(brightest_points, dtype=tf.int32)
        return brightest_points

    def pix_to_arcsec(self, brightest_points):
        """Convert pixel coordinates to arcsecond coordinates.
        Args:
            brightest_points (tf.Tensor): Tensor containing the pixel coordinates of the brightest points.
                                          Shape: (number_of_images, 2)
            num_pix (int, optional): Number of pixels in the image. Defaults to 750.
            delta_pix (float, optional): Pixel scale in arcseconds. Defaults to 0.0006.
            supersample (int, optional): Supersampling factor. Defaults to 1.
        Returns:
            tf.Tensor: Tensor containing the converted arcsecond coordinates of the brightest points.
                      Shape: (number_of_images, 2)
        """
        lo = np.arange(0, self.supersample * self.num_pixels, dtype=np.float32)
        lo = np.min(lo - np.mean(lo))
        transform_pix2angle = (tf.eye(2) * self.delta_pix) / self.supersample
        ra_at_xy_0, dec_at_xy_0 = np.squeeze((transform_pix2angle @ ([[lo], [lo]])))
        kwargs_pixel_rot = {
            "nx": self.supersample * self.num_pixels,
            "ny": self.supersample * self.num_pixels,
            "ra_at_xy_0": ra_at_xy_0,
            "dec_at_xy_0": dec_at_xy_0,
            "transform_pix2angle": np.array(transform_pix2angle),
        }
        pixel_grid_rot = PixelGrid(**kwargs_pixel_rot)
        img_x, img_y = (
            pixel_grid_rot._x_grid.astype(np.float32),
            pixel_grid_rot._y_grid.astype(np.float32),
        )
        img_x, img_y = tf.expand_dims(img_x, axis=0), tf.expand_dims(img_y, axis=0)
        img_x, img_y = tf.repeat(img_x, repeats=self.number_of_images, axis=0), tf.repeat(img_y, repeats=self.number_of_images, axis=0)
        column = tf.range(tf.shape(brightest_points)[0]) # type: ignore
        reshaped_column = tf.reshape(column, (-1, 1))
        enumerated_brightest_points = tf.concat([reshaped_column, brightest_points], axis=1)
        grid_indices, x_indices, y_indices = tf.unstack(enumerated_brightest_points, axis=-1) # type: ignore
        x_arcsec = tf.gather_nd(img_x, tf.stack([grid_indices, x_indices, y_indices], axis=-1))
        y_arcsec = tf.gather_nd(img_y, tf.stack([grid_indices, x_indices, y_indices], axis=-1))
        return x_arcsec, y_arcsec


"""
MAP
"""

def MAP(optimizer, prob_model_ps, prob_model, prob_model_uniform, posterior_version, start=None, 
        n_samples=500, num_steps=350, seed=0, track_loss=True,):
    if posterior_version == "total":
      tf.random.set_seed(seed)
      start = prob_model.prior.sample(n_samples) if start is None else start
      trial = tf.Variable(prob_model.bij.inverse(start))
    elif posterior_version == "only_dist":
      tf.random.set_seed(seed)
      start = prob_model_uniform.prior.sample(n_samples) if start is None else start
      trial = tf.Variable(prob_model_uniform.pack_bij.inverse(start))



    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            agg_loss = - prob_model_ps.log_prob(trial)

        gradients = tape.gradient(agg_loss, [trial])
        optimizer.apply_gradients(zip(gradients, [trial])) # type: ignore
        return agg_loss if track_loss else None

    losses = tf.TensorArray(tf.float32, size=0, dynamic_size=True) if track_loss else None

    with trange(num_steps) as pbar:
        for i in pbar:
            loss_value = train_step()
            if track_loss:
                losses = losses.write(i, loss_value) # type: ignore

    final_losses = losses.stack() if track_loss else None # type: ignore

    return trial, final_losses

"""
SVI
"""
def SVI(optimizer, start, n_vi, prob_model_ps, init_scales=1e-4, num_steps=500, seed=0):


        tf.random.set_seed(seed)

        start = tf.squeeze(start)
        scale = (
            np.ones(len(start)).astype(np.float32) * init_scales
            if np.size(init_scales) == 1
            else init_scales
        )
        q_z = tfd.MultivariateNormalTriL(
            loc=tf.Variable(start),
            scale_tril=tfp.util.TransformedVariable(
                np.diag(scale),
                tfp.bijectors.FillScaleTriL(diag_bijector=tfb.Exp(), diag_shift=1e-6),
                name="stddev",
            ),
        )

        losses = tfp.vi.fit_surrogate_posterior(
            lambda param: prob_model_ps.log_prob(param) ,
            surrogate_posterior=q_z,
            sample_size=n_vi,
            optimizer=optimizer,
            num_steps=num_steps,
        )


        return q_z, losses

"""
HMC
"""
def HMC(q_z,
        prob_model_ps,
        init_eps=0.3,
        init_l=3,
        n_hmc=50,
        num_burnin_steps=200,
        num_results=300,
        max_leapfrog_steps=30,
        seed=3,
  ):

      def tqdm_progress_bar_fn(num_steps):
          return iter(tqdm(range(num_steps), desc="", leave=True)) # type: ignore

      tf.random.set_seed(seed)

      mc_start = q_z.sample(n_hmc)
      cov_estimate = q_z.covariance()

      momentum_distribution = (
          tfe.distributions.MultivariateNormalPrecisionFactorLinearOperator(
              precision_factor=tf.linalg.LinearOperatorLowerTriangular(
                  tf.linalg.cholesky(cov_estimate),
              ),
              precision=tf.linalg.LinearOperatorFullMatrix(cov_estimate),
          )
      )
      @tf.function(jit_compile = True)
      def run_chain():
            num_adaptation_steps = int(num_burnin_steps * 0.8)
            start = tf.identity(mc_start)

            mc_kernel = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
                target_log_prob_fn=lambda param: prob_model_ps.log_prob(param) ,
                momentum_distribution=momentum_distribution,
                step_size=init_eps,
                num_leapfrog_steps=init_l,
            )

            mc_kernel = tfe.mcmc.GradientBasedTrajectoryLengthAdaptation(
                mc_kernel,
                num_adaptation_steps=num_adaptation_steps,
                max_leapfrog_steps=max_leapfrog_steps,
            )
            mc_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=mc_kernel, num_adaptation_steps=num_adaptation_steps
            )

            return tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=num_burnin_steps,
                current_state=start,
                kernel=mc_kernel,
                trace_fn = None,
                seed=seed,
            )

      return run_chain() # type: ignore

# testing NUTS

def HMC_nuts(q_z, prob_model_ps, init_eps=0.3, init_l=3, n_hmc=50,
        num_burnin_steps=200, num_results=300, max_leapfrog_steps=30, seed=3):

    def tqdm_progress_bar_fn(num_steps):
        return iter(tqdm(range(num_steps), desc="", leave=True)) # type: ignore

    tf.random.set_seed(seed)

    mc_start = q_z.sample(n_hmc)
    #cov_estimate = q_z.covariance()

    #momentum_distribution = tfd.MultivariateNormalLinearOperator(
    #    loc=tf.zeros(mc_start.shape[1]),  # assuming mc_start has shape [n_hmc, param_dim]
    #    scale=tf.linalg.LinearOperatorLowerTriangular(tf.linalg.cholesky(cov_estimate))
    #)

    @tf.function(jit_compile=True)
    def run_chain():
        num_adaptation_steps = int(num_burnin_steps * 0.8)
        start = tf.identity(mc_start)

        # NUTS
        nuts_kernel = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=lambda param: prob_model_ps.log_prob(param),
            step_size=init_eps,
            max_tree_depth=init_l,  # similar to num_leapfrog_steps
            #state_gradients_are_stopped=True
        )

        # step-size adaptation
        adaptive_nuts_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=nuts_kernel,
            num_adaptation_steps=num_adaptation_steps,
            #target_accept_prob=0.75
        )

        return tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=start,
            kernel=adaptive_nuts_kernel,
            trace_fn=None,
            seed=seed
        )

    return run_chain() # type: ignore


"""
Prob Model
"""
class ProbModelPS:
    def __init__(self, weight_dist, weight_flux, truth, x_arcsec, y_arcsec, prob_model, prior, observed_flux = None, flux_ratios = False,):
        self.observed_flux = tf.expand_dims(observed_flux, 1) if observed_flux is not None else None
        self.flux_ratios = flux_ratios
        self.weight_dist = weight_dist
        self.weight_flux = weight_flux
        self.truth = truth
        self.x = tf.repeat(x_arcsec[..., tf.newaxis], [1], axis=-1)
        self.y = tf.repeat(y_arcsec[..., tf.newaxis], [1], axis=-1)
        self.prior = prior
        self.prob_model = prob_model
        self.initial_setup()
        self.n = None
        #self.mag_sq_truth = None

    def initial_setup(self):
        if self.observed_flux is None:
            print("\n-simulation-")
            mag_sq_truth = tf.square(magnification(self.x, self.y, self.truth))
        else:
            mag_sq_truth = tf.constant(self.observed_flux) # input is already squared



        self.mag_sq_truth = mag_sq_truth
        #print("self.mag_sq_truth", self.mag_sq_truth)
        ratios = tf.expand_dims(mag_sq_truth, 1) / tf.expand_dims(mag_sq_truth, 0)
        mask = tf.linalg.band_part(tf.ones((4, 4), dtype=tf.bool), 0, -1)
        mask = tf.logical_and(mask, tf.logical_not(tf.linalg.diag(tf.ones(4, dtype=tf.bool))))
        upper_tri_ratios = tf.boolean_mask(ratios, mask)
        self.n = tf.shape(mag_sq_truth)[0] # type: ignore
        num_combinations = self.n * (self.n - 1) // 2
        self.ratios_truth = tf.reshape(upper_tri_ratios, [num_combinations, tf.shape(mag_sq_truth)[-1]]) # type: ignore

        i, j = tf.meshgrid(tf.range(self.n), tf.range(self.n), indexing='ij')
        upper_triangle = tf.logical_and(i < j, tf.ones_like(i, dtype=bool))
        self.i, self.j = tf.boolean_mask(i, upper_triangle), tf.boolean_mask(j, upper_triangle)

    @tf.function
    def log_prob(self, params):
        constrained = self.prob_model.bij.forward(params)

        delens = tf.convert_to_tensor(_beta_epl_shear(self.x, self.y, constrained))
        shifted_delens = tf.roll(delens, shift=-1, axis=1)
        sq_diffs = tf.square(delens - shifted_delens)
        sum_sq = tf.reduce_sum(sq_diffs, axis=0)
        sqrt_sum_sq = tf.sqrt(sum_sq)
        dist_loss = tf.reduce_mean(sqrt_sum_sq, axis=0)

        if self.flux_ratios: # flux ratios
            if self.observed_flux is None: # simulations
                print("\n--------- simulation. comparing flux ratios difference ----------")
                det_sq = tf.square(determinant(self.x, self.y, constrained))
                gi = tf.gather(det_sq, self.i, axis=0)
                gj = tf.gather(det_sq, self.j, axis=0)
                loss_terms = tf.square(gj - gi * self.ratios_truth)
                flux_ratios_loss = tf.reduce_sum(loss_terms, axis=0)
            else:
                print("\n----- real system. comparing flux ratio difference ----")
                det_sq = tf.square(determinant(self.x, self.y, constrained))
                gi = tf.gather(det_sq, self.i, axis=0)
                gj = tf.gather(det_sq, self.j, axis=0)
                loss_terms = tf.square(gj - gi * self.ratios_truth)
                flux_ratios_loss = tf.reduce_sum(loss_terms, axis=0)


        else: # magnifications
            if self.observed_flux is None: # simulations
                print("\n--------- simulation. comparing flux difference ----------")
                det_sq = tf.square(determinant(self.x, self.y, constrained))
                #print("\ndet_sq", det_sq)
                flux_diff = tf.square(det_sq - 1/self.mag_sq_truth) # type: ignore
                #print("\nflux_diff", flux_diff)
                flux_loss = tf.reduce_mean(flux_diff, axis = 0)
                #print("\nflux_loss", flux_loss)
            else:
                print("\n----- real system. comparing flux difference ----")
                det_sq = tf.square(determinant(self.x, self.y, constrained))
                #print("\ndet_sq", det_sq)
                #print("\nself.observed_flux", self.observed_flux)

                flux_diff = tf.square(det_sq * (1/15)**2 - 1/self.observed_flux) # snzwicky
                # flux_diff = tf.square(det_sq - 1/self.observed_flux)

                #print("\nflux_diff", flux_diff)
                flux_loss = tf.reduce_mean(flux_diff, axis = 0)
                #print("\nflux_loss", flux_loss)





        return - dist_loss * self.weight_dist - flux_ratios_loss * self.weight_flux + self.prior.log_prob(constrained) + self.prob_model.unconstraining_bij.forward_log_det_jacobian(self.prob_model.pack_bij.forward(params))


class ProbModelPS_only_dist:
    def __init__(self, x_arcsec, y_arcsec, prob_model):
        self.x = tf.repeat(x_arcsec[..., tf.newaxis], [1], axis=-1)
        self.y = tf.repeat(y_arcsec[..., tf.newaxis], [1], axis=-1)
        self.prob_model = prob_model

    @tf.function
    def log_prob(self, params):
        reshaped = self.prob_model.pack_bij.forward(params)
        delens = tf.convert_to_tensor(_beta_epl_shear(self.x, self.y, reshaped))
        shifted_delens = tf.roll(delens, shift=-1, axis=1)
        sq_diffs = tf.square(delens - shifted_delens)
        sum_sq = tf.reduce_sum(sq_diffs, axis=0)
        sqrt_sum_sq = tf.sqrt(sum_sq)
        dist_loss = tf.reduce_mean(sqrt_sum_sq, axis=0)

        return -dist_loss
    
class ProbModelPSsplit:
    def __init__(self, weight_dist, weight_flux, truth, x_arcsec, y_arcsec, prob_model, prior,):
        self.weight_dist = weight_dist
        self.weight_flux = weight_flux
        self.truth = truth
        self.x = tf.repeat(x_arcsec[..., tf.newaxis], [1], axis=-1)
        self.y = tf.repeat(y_arcsec[..., tf.newaxis], [1], axis=-1)
        self.prior = prior
        self.prob_model = prob_model
        self.initial_setup()
        self.n = None

    def initial_setup(self):
        mag_sq_truth = tf.square(magnification(self.x, self.y, self.truth))
        ratios = tf.expand_dims(mag_sq_truth, 1) / tf.expand_dims(mag_sq_truth, 0)
        mask = tf.linalg.band_part(tf.ones((4, 4), dtype=tf.bool), 0, -1)
        mask = tf.logical_and(mask, tf.logical_not(tf.linalg.diag(tf.ones(4, dtype=tf.bool))))
        upper_tri_ratios = tf.boolean_mask(ratios, mask)
        self.n = tf.shape(mag_sq_truth)[0] # type: ignore
        num_combinations = self.n * (self.n - 1) // 2
        self.ratios_truth = tf.reshape(upper_tri_ratios, [num_combinations, tf.shape(mag_sq_truth)[-1]]) # type: ignore

        i, j = tf.meshgrid(tf.range(self.n), tf.range(self.n), indexing='ij')
        upper_triangle = tf.logical_and(i < j, tf.ones_like(i, dtype=bool))
        self.i, self.j = tf.boolean_mask(i, upper_triangle), tf.boolean_mask(j, upper_triangle)

    @tf.function
    def log_prob_split(self, params):
        constrained = self.prob_model.bij.forward(params)

        delens = tf.convert_to_tensor(_beta_epl_shear(self.x, self.y, constrained))
        shifted_delens = tf.roll(delens, shift=-1, axis=1)
        sq_diffs = tf.square(delens - shifted_delens)
        sum_sq = tf.reduce_sum(sq_diffs, axis=0)
        sqrt_sum_sq = tf.sqrt(sum_sq)
        dist_loss = tf.reduce_mean(sqrt_sum_sq, axis=0)

        det_sq = tf.square(determinant(self.x, self.y, constrained))
        gi = tf.gather(det_sq, self.i, axis=0)
        gj = tf.gather(det_sq, self.j, axis=0)
        loss_terms = tf.square(gj - gi * self.ratios_truth)
        flux_ratios_loss = tf.reduce_sum(loss_terms, axis=0)

        return -dist_loss * self.weight_dist, - flux_ratios_loss * self.weight_flux, self.prior.log_prob(constrained) + self.prob_model.unconstraining_bij.forward_log_det_jacobian(self.prob_model.pack_bij.forward(params))

"""
Testing results
"""
class TestingResults:
    def __init__(self, truth, prob_model_output, 
                #  prior,
                #  lens_prior,
                #  phys_model,
                 num_pix = 40, delta_pix = 0.02):
        self.truth = truth
        self.prob_model_output = prob_model_output
        self.recovered_x = None
        self.recovered_y = None
        self.num_pix = num_pix
        self.delta_pix = delta_pix
        self.x_arcsec_recovered = None
        self.y_arcsec_recovered = None

        # self.prior = prior
        # self.lens_prior = lens_prior
        # self.phys_model = phys_model


    def plot_loss_evolution(self, losses):
        losses_np = losses.numpy()
        plt.figure(figsize=(6, 6))
        for i in range(10):
            plt.plot(losses_np[:, i], label=f'walker {i+1}')

        plt.title('loss evolution')
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.ylim([0, 1000])
        plt.show()

    def calculate_relative_errors(self):
        relative_errors_list = []
        for i, (truth_dict, estimated_dict) in enumerate(zip(self.truth[0], self.prob_model_output)):
            relative_errors = {}
            for key in truth_dict.keys():
                true_value = truth_dict[key]
                estimated_value = estimated_dict[key].numpy().item()
                if true_value != 0:
                    relative_error = abs((estimated_value - true_value) / true_value)
                    relative_errors[key] = relative_error
                else:
                    relative_errors[key] = float('inf')
            relative_errors_list.append(relative_errors)

        print("\nRelative errors:\n")
        for i, errors in enumerate(relative_errors_list):
            for key, error in errors.items():
                print(f"{key}: {error:.2%}")
        print("\n")

    def display_delensed_positions(self, x_arcsec, y_arcsec, _beta_epl_shear):
        x_values, y_values = _beta_epl_shear(x_arcsec, y_arcsec, [self.prob_model_output])
        x_values_np = x_values.numpy()
        y_values_np = y_values.numpy()
        self.recovered_x = np.mean(x_values_np)
        self.recovered_y = np.mean(y_values_np)
        true_x = self.truth[2][0]["center_x"]
        true_y = self.truth[2][0]["center_y"]

        relative_errors_x = (x_values_np - true_x) / true_x * 100
        relative_errors_y = (y_values_np - true_y) / true_y * 100

        plt.figure(figsize=(5, 5))
        plt.scatter(x_values_np, y_values_np, alpha=0.6, color='blue', marker='o', label='delensed points')
        plt.scatter(true_x, true_y, color='red', marker='*', s=200, label='true position')

        for i, (x, y, err_x, err_y) in enumerate(zip(x_values_np, y_values_np, relative_errors_x, relative_errors_y)):
            plt.annotate(f"err:({err_x:.2f}%, {err_y:.2f}%)", (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize = 6, color = "black")

        plt.xlabel('x coord (arcsec)')
        plt.ylabel('y coord (arcsec)')
        plt.title("delensed positions")
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

    def flux_ratio_error(self, x_arcsec, y_arcsec, mag_sq_truth):
        '''
        x_arcsec, y_arcsec are observed positions
        '''
        #print("\nla mag del input", mag_sq_truth)
        self.x = tf.repeat(x_arcsec[..., tf.newaxis], [1], axis=-1)
        self.y = tf.repeat(y_arcsec[..., tf.newaxis], [1], axis=-1)
        #####mag_de_truth = tf.square(magnification(self.x, self.y, self.truth))
        #print("\ncon truth:", mag_de_truth)

        mag_sq_truth = tf.expand_dims(mag_sq_truth, 1) if mag_sq_truth is not None else tf.square(magnification(self.x, self.y, self.truth))
        #print("\nla mag del input", mag_sq_truth)

        ratios = tf.expand_dims(mag_sq_truth, 1) / tf.expand_dims(mag_sq_truth, 0)
        #print("\nratios:", ratios)
        mask = tf.linalg.band_part(tf.ones((4, 4), dtype=tf.bool), 0, -1)
        mask = tf.logical_and(mask, tf.logical_not(tf.linalg.diag(tf.ones(4, dtype=tf.bool))))
        #print("\nmask", mask)
        upper_tri_ratios = tf.boolean_mask(ratios, mask)
        #print("\nupper tri ratios", upper_tri_ratios)
        n = tf.shape(mag_sq_truth)[0] # type: ignore
        num_combinations = n * (n - 1) // 2
        ratios_truth = tf.reshape(upper_tri_ratios, [num_combinations, tf.shape(mag_sq_truth)[-1]]) # type: ignore
        i, j = tf.meshgrid(tf.range(n), tf.range(n), indexing='ij')
        upper_triangle = tf.logical_and(i < j, tf.ones_like(i, dtype=bool))
        i, j = tf.boolean_mask(i, upper_triangle), tf.boolean_mask(j, upper_triangle)

        det_sq = tf.square(determinant(self.x, self.y, [self.prob_model_output]))
        gi = tf.gather(det_sq, i, axis=0)
        gj = tf.gather(det_sq, j, axis=0)
        loss_terms = tf.abs(gj - gi * ratios_truth)
        print("\nloss terms (in observed positions):", loss_terms)
        recovered_ratios = gj/gi
        #print("\nis j/i. j:", j)
        #print("\nis j/i. i:", i)
        #print("\nratios truth:", ratios_truth)
        #print("\nmag_sq_truth:", mag_sq_truth)

        print("\nflux ratios observed:", ratios_truth)
        print("\nflux ratios recovered (in observed positions):", gj/gi)
        rel_error = tf.abs(recovered_ratios-ratios_truth)*100/ratios_truth

        def disp_flux_error(i, j, ratios, errors):
            for idx in range(tf.shape(errors)[0]): # type: ignore
                ratio_desc = f"flux{int(i[idx])+1}/flux{int(j[idx])+1}"
                print(f"{ratio_desc}: {errors[idx][0]:.2f}%")

        #print("\nrelative errors (in observed positions):")
        disp_flux_error(i, j, ratios_truth, rel_error)

        # flux ratios recovered in recovered positions
        '''
        x_recovered = tf.repeat(self.x_arcsec_recovered[..., tf.newaxis], [chains], axis=-1)
        y_recovered = tf.repeat(self.y_arcsec_recovered[..., tf.newaxis], [chains], axis=-1)
        det_sq_rec = tf.square(determinant(x_recovered, y_recovered, [self.prob_model_output]))
        gi_rec = tf.gather(det_sq, i, axis=0)
        gj_rec = tf.gather(det_sq, j, axis=0)
        '''


    def magnification_error(self, observed_magnifications,):
        if observed_magnifications is not None:
          #print("\nobserved_magnifications:", tf.sqrt(observed_magnifications))
          print("\nobserved_magnifications SN Zicky:", tf.sqrt(observed_magnifications)/15)
        x_rec = tf.repeat(self.x_arcsec_recovered[..., tf.newaxis], [1], axis=-1) # type: ignore
        y_rec = tf.repeat(self.y_arcsec_recovered[..., tf.newaxis], [1], axis=-1) # type: ignore
        predicted_mag_rec = magnification(x_rec, y_rec, [self.prob_model_output])
        print("\nPredicted magnifications (in recovered positions). The ordering may have changed:", predicted_mag_rec)

        predicted_mag = magnification(self.x, self.y, [self.prob_model_output])
        print("\nPredicted magnifications (in observed positions):", predicted_mag)



    def relens(self, x_arcsec, y_arcsec,
               prior, lens_prior, phys_model):
        #delta_pix = self.delta_pix
        supersample = 1
        #num_pix = self.num_pix

        prob_model = ForwardProbModel(prior, 0, background_rms=0.2, exp_time=100)
        example = lens_prior.sample(seed = 0)
        size = int(tf.size(tf.nest.flatten(example))) # type: ignore

        sim_config = SimulatorConfig(delta_pix=self.delta_pix, num_pix=self.num_pix, supersample=supersample)
        lens_sim = LensSimulator(phys_model, sim_config, bs=1)


        print("self.prob_model_output", self.prob_model_output)
        lens_omitted_img = lens_sim.simulate([self.prob_model_output,[],[{"center_x": self.recovered_x, "center_y": self.recovered_y, 'Ie': 2815.97}]]) # Omit lens light
        converter = BrightestPoints(number_of_images = 4, num_pixels=self.num_pix, grid_size=50, delta_pix=self.delta_pix, supersample=1)
        plt.imshow(lens_omitted_img) # type: ignore
        plt.show()
        print(lens_omitted_img)
        brightest_pix_recovered = converter.find_brightest_points(lens_omitted_img)
        #plt.plot(brightest_pixels[:, 1], brightest_pixels[:, 0], ".", ms = 15, color = "dodgerblue", alpha = 0.6, label = "brighest points")
        plt.title("selected brightest points")
        plt.legend()
        self.x_arcsec_recovered, self.y_arcsec_recovered = converter.pix_to_arcsec(brightest_pix_recovered)
        chains = 1
        x = tf.repeat(self.x_arcsec_recovered[..., tf.newaxis], [chains], axis=-1)
        y = tf.repeat(self.y_arcsec_recovered[..., tf.newaxis], [chains], axis=-1)
        print("Brighest points reshaped into x, y:")
        print("\nx:", x)
        print("\ny:", y)

        # truth

        #truth_img = lens_sim.simulate([self.truth[0], [], self.truth[2]])
        #converter = BrightestPoints(number_of_images=4, num_pixels=self.num_pix, grid_size=20, delta_pix=self.delta_pix, supersample=1)
        #brightest_pix_truth = converter.find_brightest_points(truth_img)
        #x_arcsec_truth, y_arcsec_truth = converter.pix_to_arcsec(brightest_pix_truth)

        params = self.prob_model_output

        print("\nparams", params)

        kwargs_main_lens = {
            'theta_E': params[0]['theta_E'].numpy()[0],
            'gamma': params[0]['gamma'].numpy()[0],
            'e1': params[0]['e1'].numpy()[0],
            'e2': params[0]['e2'].numpy()[0],
            'center_x': params[0]['center_x'].numpy()[0],
            'center_y': params[0]['center_y'].numpy()[0],}
        kwargs_shear = {
            'gamma1': params[1]['gamma1'].numpy()[0],
            'gamma2': params[1]['gamma2'].numpy()[0],
        }
        kwargs_lens = [kwargs_main_lens, kwargs_shear]
        print("\nkwargs_lens", kwargs_lens)

        lens_model = LensModel(lens_model_list=['EPL', 'SHEAR'])
        fig, axes = plt.subplots(figsize=(12, 12))
        extent = [-self.num_pix / 2 * self.delta_pix, self.num_pix / 2 * self.delta_pix, - self.num_pix / 2 * self.delta_pix, self.num_pix / 2 * self.delta_pix]

        lens_plot.lens_model_plot(axes, lensModel=lens_model, kwargs_lens=kwargs_lens, numPix=self.num_pix, deltaPix=self.delta_pix,
                                  sourcePos_x=self.recovered_x, sourcePos_y=self.recovered_y, point_source=True, with_caustics=True, # type: ignore
                                  fast_caustic=False, coord_inverse=True) 
        axes.imshow(lens_omitted_img, origin='lower', vmin=0, vmax=10, extent=extent) # type: ignore
        axes.plot(x_arcsec, y_arcsec, '.', color="blue", ms=15, label = "observed positions")
        axes.plot(self.x_arcsec_recovered, self.y_arcsec_recovered, '.', color = "red", ms = 15, label = "recovered positions")
        #axes.plot(self.truth[2][0]['center_x'], self.truth[2][0]['center_y'], ".", color="yellow", ms=15)
        axes.plot(self.recovered_x, self.recovered_y, ".", color="yellow", ms=15, label = "recovered source") #type: ignore
        for text in axes.texts:
          text.set_color('white')

        axes.legend(fontsize=14)
        axes.grid(False)
        axes.invert_yaxis()

        for line in axes.lines:
           if line.get_marker() == 'd':
            line.set_markerfacecolor('lightgray')
            line.set_markeredgecolor('gray')
            line.set_alpha(0.5)

        plt.show()

# test_results = TestingResults(truth, prob_model_output)
# test_results.plot_loss_evolution(losses)
# test_results.calculate_relative_errors()
# test_results.display_delensed_positions(x_arcsec, y_arcsec, _beta_epl_shear)
# test_results.flux_ratio_error()

"""
Pipeline
"""
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Plots import lens_plot
import corner

# todo: include no prior modeling
# todo: show relative values

class LensModelAnalysis:
    def __init__(self, delta_pix, num_pix, truth_test, prob_model, prob_model_uniform, prior,
                 lens_prior, phys_model,
                 observed_data = None, weight_dist =  1.*1e3, weight_flux = 1.*1e2, simulation = False,
                 flux_ratios = False):
        self.simulation = simulation
        self.delta_pix = delta_pix
        self.num_pix = num_pix
        self.weight_dist = weight_dist
        self.weight_flux = weight_flux
        self.truth_test = truth_test
        self.prob_model = prob_model
        self.prob_model_uniform = prob_model_uniform
        self.prior = prior
        self.lens_model = LensModel(lens_model_list=['EPL', 'SHEAR'])
        self.best = None
        self.q_z = None
        self.observed_data = observed_data # a list. the first sublist are observed positions, [[x] [y]]; the second are observed magnifications
        self.flux_ratios = flux_ratios

        # fixes
        self.lens_prior = lens_prior
        self.phys_model = phys_model



    def run_map(self, 
                lens_sim,
                n_map = 4000, n_steps = 2000,):
        if self.simulation:
            print("\nSimulated image and brightest points:\n")
            lens_omitted_img = lens_sim.simulate([self.truth_test[0], [], self.truth_test[2]])
            converter = BrightestPoints(number_of_images=4, num_pixels=self.num_pix, grid_size=20, delta_pix=self.delta_pix, supersample=1) # 40
            #print("lens_omitted_img", lens_omitted_img.shape)
            brightest_pixels = converter.find_brightest_points(lens_omitted_img)
            #print("brightest_pixels", brightest_pixels.shape)
            self.x_arcsec, self.y_arcsec = converter.pix_to_arcsec(brightest_pixels)
            self.observed_flux = None
        else:
            print("\n--- Modeling a real system ----")
            self.x_arcsec, self.y_arcsec = self.observed_data[0][0], self.observed_data[0][1] # type: ignore
            self.observed_flux = self.observed_data[1] # type: ignore


        '''
        fig, axes = plt.subplots(figsize=(10, 8))
        lens_plot.lens_model_plot(axes, lensModel=self.lens_model, kwargs_lens=[self.truth_test[0][0], self.truth_test[0][1]], 
                                  numPix=self.num_pix, deltaPix=self.delta_pix, sourcePos_x=self.truth_test[2][0]['center_x'], 
                                  sourcePos_y=self.truth_test[2][0]['center_y'], point_source=True, with_caustics=True, 
                                  fast_caustic=False, coord_inverse=False)
        axes.plot(self.x_arcsec, self.y_arcsec, '.', color="blue", ms=15)
        axes.plot(self.truth_test[2][0]['center_x'], self.truth_test[2][0]['center_y'], ".", color="yellow", ms=15)
        axes.grid(False)
        axes.invert_yaxis()
        plt.show()'''

        prob_model_ps = ProbModelPS(weight_dist = self.weight_dist, weight_flux = self.weight_flux, truth = self.truth_test, 
                                    x_arcsec = self.x_arcsec, y_arcsec = self.y_arcsec, prob_model = self.prob_model, 
                                    prior = self.prior, observed_flux = self.observed_flux, flux_ratios = self.flux_ratios)
        schedule_fn = tf.keras.optimizers.schedules.PolynomialDecay(1e-1, n_steps, 1e-2/5) # type: ignore
        optimizer = tf.keras.optimizers.Adam(schedule_fn) # type: ignore
        MAP_sample, losses = MAP(posterior_version = "total", optimizer=optimizer, n_samples=n_map, num_steps=n_steps, seed=0, 
                                 prob_model_ps = prob_model_ps, prob_model = self.prob_model, prob_model_uniform = self.prob_model_uniform)
        lps = prob_model_ps.log_prob(MAP_sample)
        self.best = MAP_sample[tf.argmax(lps)] # type: ignore
        #print("best GD format", self.best)

        prob_model_output = self.prob_model.bij.forward([self.best])[0]
        print("\nBest parameters:\n", prob_model_output)
        test_results = TestingResults(self.truth_test, prob_model_output)
        test_results.plot_loss_evolution(losses)
        test_results.calculate_relative_errors()
        test_results.display_delensed_positions(self.x_arcsec, self.y_arcsec, _beta_epl_shear)
        ### test_results.flux_ratio_error(self.x_arcsec, self.y_arcsec)

        # check gamma value when only the distance term is included with no prior
        print("\nresults when only the distance term is included in the loss with no prior --------------------------------------------------\n")
        prob_model_only_dist = ProbModelPS_only_dist(x_arcsec = self.x_arcsec, y_arcsec = self.y_arcsec, prob_model = self.prob_model_uniform,)
        schedule_fn = tf.keras.optimizers.schedules.PolynomialDecay(1e-1, n_steps, 1e-2/5) # type: ignore
        optimizer = tf.keras.optimizers.Adam(schedule_fn) # type: ignore
        MAP_sample, losses = MAP(posterior_version = "only_dist", optimizer=optimizer, n_samples=n_map, num_steps=n_steps, seed=0, 
                                 prob_model_ps = prob_model_only_dist, prob_model = self.prob_model, prob_model_uniform = self.prob_model_uniform)
        lps = prob_model_only_dist.log_prob(MAP_sample)
        best = MAP_sample[tf.argmax(lps)] # type: ignore
        prob_model_output_dist = self.prob_model_uniform.pack_bij.forward([best])[0]
        print("\nBest parameters (only distance term):\n", prob_model_output_dist)

        test_results_dist = TestingResults(self.truth_test, prob_model_output_dist)
        test_results_dist.calculate_relative_errors()
        test_results_dist.display_delensed_positions(self.x_arcsec, self.y_arcsec, _beta_epl_shear)
        ### test_results_dist.flux_ratio_error(self.x_arcsec, self.y_arcsec)


    def run_vi(self, n_vi = 500, num_steps = 600):
        prob_model_ps = ProbModelPS(weight_dist = self.weight_dist, weight_flux = self.weight_flux, truth = self.truth_test, 
                                    x_arcsec = self.x_arcsec, y_arcsec = self.y_arcsec, prob_model = self.prob_model, 
                                    prior = self.prior, observed_flux = self.observed_flux, flux_ratios = self.flux_ratios)
        # TODO: CHECK POSSIBLE ERROR below set not function
        schedule_fn = tf.keras.optimizers.schedules.PolynomialDecay(0.0, 500, 4e-2, 1.0) # type: ignore
        optimizer = tf.keras.optimizers.Adam(schedule_fn) # type: ignore
        self.q_z, losses_vi = SVI(optimizer=optimizer, start=self.best, n_vi=n_vi, num_steps=num_steps, prob_model_ps = prob_model_ps)

        plt.plot(losses_vi)
        plt.title("ELBO loss")
        plt.grid(True)
        plt.show()


    def run_hmc(self, n_hmc = 50, num_burnin_steps = 100, num_results = 750, method = "adaptative"):
        if method == "adaptative":
            prob_model_ps = ProbModelPS(weight_dist = self.weight_dist, weight_flux = self.weight_flux, truth = self.truth_test, 
                                        x_arcsec = self.x_arcsec, y_arcsec = self.y_arcsec, prob_model = self.prob_model, 
                                        prior = self.prior, observed_flux = self.observed_flux, flux_ratios = self.flux_ratios)
            samples = HMC(q_z=self.q_z, n_hmc=n_hmc, init_eps=0.5, init_l=3, max_leapfrog_steps=100, 
                          num_burnin_steps=num_burnin_steps, num_results=num_results, prob_model_ps = prob_model_ps)

        if method == "nuts":
            prob_model_ps = ProbModelPS(weight_dist = self.weight_dist, weight_flux = self.weight_flux, truth = self.truth_test, 
                                        x_arcsec = self.x_arcsec, y_arcsec = self.y_arcsec, prob_model = self.prob_model, prior = self.prior)
            samples = HMC_nuts(q_z=self.q_z, n_hmc=n_hmc, init_eps=0.5, init_l=3, max_leapfrog_steps=100, num_burnin_steps=num_burnin_steps, 
                               num_results=num_results, prob_model_ps = prob_model_ps)

        print(f"\n----------------method: {method}------------------\n")
        get_samples = lambda x: tf.convert_to_tensor([
            x[0][0]['theta_E'],
            x[0][0]['gamma'],
            x[0][0]['e1'],
            x[0][0]['e2'],
            x[0][0]['center_x'],
            x[0][0]['center_y'],
            x[0][1]['gamma1'],
            x[0][1]['gamma2'],
        ])
        physical_samples = get_samples(self.prob_model.bij.forward(samples)).numpy() # type: ignore

        parameter_names = ['theta_E', 'gamma', 'e1', 'e2', 'center_x', 'center_y', 'gamma1', 'gamma2']
        n_params = len(parameter_names)
        n_cols = 2
        n_rows = (n_params + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2 * n_rows))
        axes = axes.flatten()
        for i, param_name in enumerate(parameter_names):
            ax = axes[i]
            ax.plot(physical_samples[i, :, 0:6], label=f'{param_name}', alpha=0.4)
            ax.set_title(f'{param_name}')
            ax.set_xlabel('Steps')
            ax.grid(False)
        fig.tight_layout()
        plt.show()

        Rhat = tfp.mcmc.potential_scale_reduction(samples).numpy()
        ESS = tfp.mcmc.effective_sample_size(samples, cross_chain_dims=1).numpy()
        self.print_formatted_values(self.prob_model.pack_bij.forward([Rhat])[0], "rhat")
        self.print_formatted_values(self.prob_model.pack_bij.forward([ESS])[0], "ess")

        markers = get_samples(self.truth_test)
        plt.figure(figsize=(14, 14))
        fig = corner.corner(physical_samples.reshape((8,-1)).T,
                            show_titles=True, title_fmt='.3f',
                            labels=[r'$\theta_E$', r'$\gamma$', r'$\epsilon_1$', r'$\epsilon_2$', r'$x$', r'$y$', 
                                    r'$\gamma_{1,ext}$', r'$\gamma_{2,ext}$'],fig=plt.gcf());
        plt.show()

        mean_values = {}
        median_values = {}
        median_list = []
        for i, param_name in enumerate(parameter_names):
            param_data = physical_samples[i, :]
            mean_values[param_name] = np.mean(param_data)
            median_values[param_name] = np.median(param_data)
            median_list.append(np.median(param_data))
        median_list = tf.convert_to_tensor(median_list)


        print("Parameter Means and Medians:")
        for param_name in parameter_names:
            print(f"{param_name}: mean = {mean_values[param_name]:.3f}, median = {median_values[param_name]:.3f}")


        dict1_keys = {'theta_E', 'gamma', 'e1', 'e2', 'center_x', 'center_y'}
        dict2_keys = {'gamma1', 'gamma2'}

    # Splitting the dictionary and creating lists of dictionaries
        list_of_dicts1 = [{key: tf.convert_to_tensor([median_values[key]]) for key in dict1_keys}]
        list_of_dicts2 = [{key: tf.convert_to_tensor([median_values[key]]) for key in dict2_keys}]
        #print("list of dicts 1", list_of_dicts1)
        #print("list of dicts 2", list_of_dicts2)
        combined = [list_of_dicts1[0], list_of_dicts2[0]]

        #print("combined", combined)
        #median_values_list = tf.reshape(combined, [8])

        #print("median values hmc format", combined)
        testin = self.prob_model.pack_bij.inverse([combined])
        testin = tf.reshape(testin, [8])

        median_hmc_output = self.prob_model.pack_bij.forward([testin])[0]

        #print("\nmedian hmc:\n", median_hmc_output)
        test_results = TestingResults(self.truth_test, median_hmc_output, num_pix = self.num_pix, delta_pix = self.delta_pix)
        #test_results.calculate_relative_errors()
        test_results.display_delensed_positions(self.x_arcsec, self.y_arcsec, _beta_epl_shear)
        ###test_results.flux_ratio_error(self.x_arcsec, self.y_arcsec)
        test_results.relens(self.x_arcsec, self.y_arcsec,
                            self.prior, self.lens_prior, self.phys_model)
        test_results.flux_ratio_error(self.x_arcsec, self.y_arcsec, self.observed_flux)
        test_results.magnification_error(observed_magnifications = self.observed_flux)


        # relative weights
        '''
        samples_split = tf.reshape(samples, [-1, 8])
        rel_weights_logprob = ProbModelPSsplit(weight_dist = self.weight_dist, weight_flux = self.weight_flux, truth = self.truth_test, 
                                               x_arcsec = self.x_arcsec, y_arcsec = self.y_arcsec, prob_model = prob_model, prior = prior)
        rel_weights = rel_weights_logprob.log_prob_split(samples_split)


        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = ['distance loss', 'flux loss', 'prior loss']

        # Iterate through each tensor, axis, and title
        for ax, tensor, title in zip(axes, rel_weights, titles):

            data = tensor.numpy()


            mean_val = np.mean(data)


            ax.hist(data, bins=50, color='skyblue', alpha=0.7)


            ax.set_title(f"{title} (mean: {mean_val:.2f})")


            ax.set_xlabel('value')



        plt.tight_layout()


        plt.show()'''
        return self.prob_model.bij.forward(samples)



    def print_formatted_values(self, output, variable = "rhat"):
        print("\nRhat:") if variable == "rhat" else print("\nESS:")
        for index, param_dict in enumerate(output):
            for key, tensor in param_dict.items():
                numeric_value = tensor.numpy()[0]
                print(f"{key}: {numeric_value:.6f}")

