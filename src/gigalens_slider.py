import gigalens
from gigalens.jax.inference import ModellingSequence
from gigalens.jax.model import ForwardProbModel, BackwardProbModel
from gigalens.model import PhysicalModel
from gigalens.jax.simulator import LensSimulator
from gigalens.simulator import SimulatorConfig
from gigalens.jax.profiles.light import sersic, shapelets
from gigalens.jax.profiles.mass import epl, shear
import lenstronomy
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Plots import lens_plot
import tensorflow_probability.substrates.jax as tfp
import jax
from jax import random
import numpy as np
import optax
from jax import numpy as jnp
from matplotlib import pyplot as plt

tfd = tfp.distributions

import lenstronomy
import copy
import numpy as np
import imageio
import os
from matplotlib import pyplot as plt

from matplotlib.widgets import Slider

from lenstronomy.SimulationAPI.ObservationConfig.Roman import Roman
from lenstronomy.SimulationAPI.sim_api import SimAPI
import lenstronomy.Plots.plot_util as plot_util
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.Plots import lens_plot
import lenstronomy.Util.param_util as param_util
from lenstronomy.Util import util
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from os.path import expanduser
import sys
import os
from pathlib import Path

from matplotlib import gridspec

#sys.path.insert(0, '/Users/cltan/celine-gigalens/gigalens/src')

def image_sim(theta_E, q, phi,lens_z,lra,ldec,source_z, sra,sdec,lens_light, gamma, gamma_ext, psi_ext, source_light=True,ps_light=False):
#     theta_E: Einstein radius
#     q: axial ratio
#     phi: position angle
#     lens_z: lens redshift    
#     lra: lens right ascension
#     ldec: lens declination
#     source_z: source redshift
#     sra: source right ascension
#     sdec: source declination
#     lens_light:
#     gamma: 
#     gamma_ext: 
#     psi_ext:
#     source_light:
#     ps_light:
    
    # Convert shear strength and angle to gamma_1 and gamma_2
    # similar to Eq 5 in paper
    gamma_1 = gamma_ext * np.cos(2 * psi_ext)
    gamma_2 = gamma_ext * np.sin(2 * psi_ext)
    
    # loading simulation filters
    Roman_g = lenstronomy.SimulationAPI.ObservationConfig.Roman.Roman(band='F062', psf_type='PIXEL', survey_mode='wide_area')
    Roman_r = lenstronomy.SimulationAPI.ObservationConfig.Roman.Roman(band='F106', psf_type='PIXEL', survey_mode='wide_area')
    Roman_i = lenstronomy.SimulationAPI.ObservationConfig.Roman.Roman(band='F184', psf_type='PIXEL', survey_mode='wide_area')
    roman = [Roman_g, Roman_r, Roman_i]
    
    band_b, band_g, band_r = roman
    kwargs_b_band = band_b.kwargs_single_band()
    kwargs_g_band = band_g.kwargs_single_band()
    kwargs_r_band = band_r.kwargs_single_band()
    '''
    
    '''
    kwargs_model = {'lens_model_list': ['EPL', 'SHEAR'], # list of lens models to be used
                    'lens_redshift_list': [lens_z, lens_z],
                    'z_source': source_z,
                    'lens_light_model_list': ['SERSIC_ELLIPSE'],  # list of unlensed light models to be used
                    'source_light_model_list': ['SERSIC'],  # list of extended source models to be used, here we used the interpolated real galaxy
                    }
    # to obtain ellipticity from orientation angle and axial ratio
    e1,e2 = param_util.phi_q2_ellipticity(q=q,phi=phi)
    #e1,e2 = e1,e2
    
    # lens model parameters
    kwargs_lens = [
    {'theta_E': theta_E, 'center_x': lra, 'center_y': ldec, 'e1':e1 , 'e2':e2, 'gamma':gamma},
    {'gamma1': gamma_1, 'gamma2': gamma_2}
    ]

    # lens light
    kwargs_lens_light_mag_g = [{'magnitude': 23, 'R_sersic': 1., 'n_sersic': 1.01, 'center_x': lra+0.031, 'center_y': ldec+0.019,'e1':0.2,'e2':-0.1}]
    # source light
    kwargs_source_mag_g = [{'magnitude': 28, 'center_x': sra, 'center_y': sdec,'R_sersic': 0.1, 'n_sersic': 2}]
    # point source
    kwargs_ps_mag_g = [{'magnitude': 2, 'ra_source': 0, 'dec_source': 0}]
    # 32**2 * 2 * 2 * 2 (each filter)
    numpix = 1024 * 8  # number of pixels per axis of the image to be modelled

    kwargs_numerics = {'point_source_supersampling_factor': 1}

    size = 10. # width of the image in units of arc seconds


    # r-band
    g_r_source = 1  # color mag_g - mag_r for source
    g_r_lens = -1  # color mag_g - mag_r for lens light
    g_r_ps = 0
    kwargs_lens_light_mag_r = copy.deepcopy(kwargs_lens_light_mag_g)
    kwargs_lens_light_mag_r[0]['magnitude'] -= g_r_lens

    kwargs_source_mag_r = copy.deepcopy(kwargs_source_mag_g)
    kwargs_source_mag_r[0]['magnitude'] -= g_r_source

    kwargs_ps_mag_r = copy.deepcopy(kwargs_ps_mag_g)
    kwargs_ps_mag_r[0]['magnitude'] -= g_r_ps
    #print(kwargs_ps_mag_r[0]['amp'])

    # i-band
    g_i_source = 2
    g_i_lens = -2
    g_i_ps = 0
    kwargs_lens_light_mag_i = copy.deepcopy(kwargs_lens_light_mag_g)
    kwargs_lens_light_mag_i[0]['magnitude'] -= g_i_lens

    kwargs_source_mag_i = copy.deepcopy(kwargs_source_mag_g)
    kwargs_source_mag_i[0]['magnitude'] -= g_i_source

    kwargs_ps_mag_i = copy.deepcopy(kwargs_ps_mag_g)
    kwargs_ps_mag_i[0]['magnitude'] -= g_i_ps

    pixel_scale = kwargs_g_band['pixel_scale']
    numpix = int(round(size / pixel_scale))

    # Lenstronomy simulation with model
    sim_b = lenstronomy.SimulationAPI.sim_api.SimAPI(numpix=numpix, kwargs_single_band=kwargs_b_band, kwargs_model=kwargs_model)
    sim_g = lenstronomy.SimulationAPI.sim_api.SimAPI(numpix=numpix, kwargs_single_band=kwargs_g_band, kwargs_model=kwargs_model)
    sim_r = lenstronomy.SimulationAPI.sim_api.SimAPI(numpix=numpix, kwargs_single_band=kwargs_r_band, kwargs_model=kwargs_model)

    kwargs_numerics = {'point_source_supersampling_factor': 1}

    # return the ImSim instance. With this class instance, you can compute all the
    # modelling accessible of the core modules. See class documentation and other notebooks.
    imSim_b = sim_b.image_model_class(kwargs_numerics)
    imSim_g = sim_g.image_model_class(kwargs_numerics)
    imSim_r = sim_r.image_model_class(kwargs_numerics)

    # turn magnitude kwargs into lenstronomy kwargs
    kwargs_lens_light_g, kwargs_source_g, kwargs_ps_g = sim_b.magnitude2amplitude(kwargs_lens_light_mag_g, kwargs_source_mag_g, kwargs_ps_mag_g)
    kwargs_lens_light_r, kwargs_source_r, kwargs_ps_r = sim_g.magnitude2amplitude(kwargs_lens_light_mag_r, kwargs_source_mag_r, kwargs_ps_mag_r)
    kwargs_lens_light_i, kwargs_source_i, kwargs_ps_i = sim_r.magnitude2amplitude(kwargs_lens_light_mag_i, kwargs_source_mag_i, kwargs_ps_mag_i)
    #print(kwargs_source_i)

    params = ([
    {'theta_E': theta_E, 
     'gamma': gamma, 
     'e1': e1, 
     'e2': e2, 
     'center_x': lra+0.031, 
     'center_y': ldec+0.019},
    {'gamma1': gamma_1, 
     'gamma2': gamma_2} ], 
    [{'R_sersic': 1., 
     'n_sersic': 1.01, 
     'e1': e1, 
     'e2': e2, 
     'center_x': lra+0.031, 
     'center_y': ldec+0.019, 
     'Ie': kwargs_lens_light_i[0]['amp'] + kwargs_lens_light_g[0]['amp'] + kwargs_lens_light_r[0]['amp']}], # ask about these parameters
    
    [{'R_sersic': 0.1, 
      'n_sersic': 2., 
      'e1': e1, 
      'e2': e2, 
      'center_x': sra, 
      'center_y': sdec, 
      'Ie': kwargs_source_i[0]['amp'] + kwargs_source_g[0]['amp'] + kwargs_source_r[0]['amp']}])


    image_b = imSim_b.image(kwargs_lens, kwargs_source_g, kwargs_lens_light_g, kwargs_ps_g,lens_light_add=lens_light,source_add=source_light,point_source_add=source_light)
    image_g = imSim_g.image(kwargs_lens, kwargs_source_r, kwargs_lens_light_r, kwargs_ps_r,lens_light_add=lens_light,source_add=source_light,point_source_add=source_light)
    image_r = imSim_r.image(kwargs_lens, kwargs_source_i, kwargs_lens_light_i, kwargs_ps_i,lens_light_add=lens_light,source_add=source_light,point_source_add=source_light)

    img = np.zeros((image_g.shape[0], image_g.shape[1], 3), dtype=float)
    scale_max=0.1
    def _scale_max(image):
        flat=image.flatten()
        flat.sort()
        scale_max = flat[int(len(flat)*0.95)]
        return scale_max
    img[:,:,0] = lenstronomy.Plots.plot_util.sqrt(image_b, scale_min=0, scale_max=scale_max)
    img[:,:,1] = lenstronomy.Plots.plot_util.sqrt(image_g, scale_min=0, scale_max=scale_max)
    img[:,:,2] = lenstronomy.Plots.plot_util.sqrt(image_r, scale_min=0, scale_max=scale_max)
    data_class = sim_b.data_class

    # caustics and critical curve
    pixel_scale = kwargs_g_band['pixel_scale']
    numpix = int(round(size / pixel_scale))
    deltaPix = pixel_scale
    # Robust way to get the path relative to this file
    # base_path = Path(__file__).parent.parent  # adjust as needed
    # psf_path = base_path / 'gigalens' / 'src' / 'gigalens' / 'assets' / 'psf.npy'
    kernel = np.load('gigalens/src/gigalens/assets/psf.npy').astype(np.float32)
    phys_model = PhysicalModel([epl.EPL(50), shear.Shear()], [sersic.SersicEllipse(use_lstsq=False)], [sersic.SersicEllipse(use_lstsq=False)])
    sim_config = SimulatorConfig(delta_pix=deltaPix, num_pix=numpix, supersample=1, kernel=kernel)
    lens_sim = LensSimulator(phys_model, sim_config, bs=1)

    
    sim_lens = lens_sim.simulate(params)

    kwargs_data = lenstronomy.Util.simulation_util.data_configure_simple(numpix*2, deltaPix)
    data = lenstronomy.Data.imaging_data.ImageData(**kwargs_data)
    _coords = data

    lens_model_list = kwargs_model['lens_model_list']
    lensModel = lenstronomy.LensModel.lens_model.LensModel(lens_model_list=lens_model_list)

    '''
    x, y = np.meshgrid(
        np.linspace(-numpix * deltaPix / 2, numpix * deltaPix / 2, int(numpix / 2)),
        np.linspace(-numpix * deltaPix / 2, numpix * deltaPix / 2, int(numpix / 2))
    )'''
    x, y = util.make_grid(numPix=numpix, deltapix=deltaPix)
    #kappa = lensModel.kappa(x, y, kwargs=kwargs_lens)
    #gamma1, gamma2 = lensModel.gamma(x, y, kwargs=kwargs_lens)
    #shear = np.sqrt(gamma1**2 + gamma2**2)
    #magnification = lensModel.magnification(x, y, kwargs=kwargs_lens)
    #mag_log = np.log(np.abs(magnification))
    #mag_img = util.array2image(mag_log)
    x_grid, y_grid = np.meshgrid(
        np.linspace(-numpix * deltaPix / 2, numpix * deltaPix / 2, int(numpix)),
        np.linspace(-numpix * deltaPix / 2, numpix * deltaPix / 2, int(numpix))
    )

    #x, y = util.make_grid(numPix=numpix, deltapix=deltaPix)
    kappa = lensModel.kappa(x_grid, y_grid, kwargs_lens)
    g_1, g_2 = lensModel.gamma(x_grid, y_grid, kwargs_lens)
    #shear = np.sqrt(gamma1**2 + gamma2**2)
    det_A = (1 - kappa)**2 - (g_1**2 + g_2**2)
    #magnification = lensModel.magnification(x, y, kwargs=kwargs_lens)
    #mag_log = np.log(np.abs(magnification))
    #mag_img = util.array2image(1 / det_A)
    mag_img = 1/det_A


    return sim_lens, kwargs_lens, lensModel, _coords, mag_img, img
'''
def evolve(theta_E,e1,e2,lens_z,lra,ldec,source_z,sra,sdec,lens_light,source_light=True,ps_light=False):
    function_arguments = locals()
    def customLength(input):
        #returns length of object if it's a list or array, otherwise returns 1
        try:
            return len(input)
        except:
            return 1
    
    maxLength = max([customLength(function_arguments[i]) for i in function_arguments.keys()])
    
    #checks if each the length of each argument is equal to the max of the length of all arguments or equal to 1. If the length is 1 we make it equal to the max length
    
    for ii in function_arguments.keys():
        if customLength(function_arguments[ii]) == 1:
            function_arguments[ii]=[function_arguments[ii]] * maxLength
        elif customLength(function_arguments[ii]) != maxLength: 
            raise Exception(ii+": all arguments of evolve() must have the same length (>1), or length 1. The maximum length of the arguments of evolve() is "+str(maxLength)+", but "+ii+" has length "+str(customLength(function_arguments[ii]))+".")
            
    output = []
    for i, (theta_E,e1,e2,lens_z,lra,ldec,source_z,sra,sdec,lens_light,source_light,ps_light) in enumerate(zip(*[function_arguments[ii] for ii in function_arguments.keys()])):
        
        img, kwargs_lens, lensModel, _coords  = image_sim(theta_E = theta_E, e1=e1, e2=e2, lens_z=lens_z, lra = ldec,ldec = lra, source_z=source_z, sra = sra, sdec = sdec, lens_light = True, source_light = True, ps_light = True)
        
        outputStep = {"img": img,"kwargs_lens": kwargs_lens,"lensModel": lensModel,"_coords": _coords,"lra": lra,"ldec": ldec,"sra": sra,"sdec": sdec}
        output.append(outputStep)
    
    return output


def animate(input):
    for i in range(0,len(input)):
        img, kwargs_lens, lensModel, _coords, lra, ldec, sra, sdec = [input[i][iii] for iii in input[i].keys()]

        fig,ax = plt.subplots(1,1,figsize = (7,7))
        lenstronomy.Plots.lens_plot.caustics_plot(ax, _coords, lensModel, kwargs_lens, fast_caustic=True, color_crit='red', color_caustic='green', alpha = 0.6)

        ax.imshow(img, aspect='equal', origin='lower', extent=[-5, 5, -5, 5])
        ax.plot(sra, sdec, 'ro')

        plt.savefig(f'animation_frames/img-{i}.png')
        plt.close()
            
    with imageio.get_writer('animation.gif', mode='V', format = 'GIF', duration = 30) as writer:
        for i in range(len(input)):
            image = imageio.imread(f'animation_frames/img-{i}.png')
            writer.append_data(image)

'''

def lenstronomy_sqrt_rescale(image, scale_min=0, scale_max=0.1):
    imageData = np.array(image, copy=True)
    imageData = np.clip(imageData, scale_min, scale_max)
    
    imageData -= scale_min
    imageData[imageData < 0] = 0.0
    
    imageData = np.sqrt(imageData)
    
    dynamic_range = scale_max - scale_min
    imageData /= np.sqrt(dynamic_range)
    
    return imageData

def sqrt_rescale(image, scale_min=0, scale_max=None):
    if scale_max is None:
        flat = image.flatten()
        flat.sort()
        scale_max = flat[int(len(flat) * 0.95)]
    
    scaled_image = np.sqrt(np.clip(image - scale_min, 0, None)) / np.sqrt(scale_max - scale_min)
    
    return scaled_image
# initial parameters
theta_E_init = 1.5
# e1_init = 0.1
# e2_init = -0.1
q_init = 0.7
phi_init = 0.0
lens_z = 0.5
lra = 0.0
ldec = 0.0
source_z = 1.5
sra_init = 0.1
sdec_init = 0.1
lens_light = True
source_light = True
ps_light = True
gamma_init = 2.0
#gamma_1_init = 0
#gamma_2_init = 0.05
gamma_ext_init = 0.05
psi_ext_init = 0.0

# initial image and plot
img, kwargs_lens, lens_model, coords, mag, lensstron_img = image_sim(
    theta_E_init, q_init, phi_init, lens_z, lra, ldec, source_z, sra_init, sdec_init, lens_light, gamma_init, gamma_ext_init, psi_ext_init, source_light, ps_light
)

'''
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.6)
image_display = ax.imshow(img, aspect='equal', origin='lower', extent=[-5, 5, -5, 5])
lens_plot.caustics_plot(ax, coords, lens_model, kwargs_lens, fast_caustic=True, color_crit='red', color_caustic='green', alpha=0.6)
ax.plot(sra_init, sdec_init, 'bx')

# text
text_kappa = ax.text(0.05, 0.95, '', transform=ax.transAxes, color="white", fontsize=5, va='top')
text_gamma = ax.text(0.05, 0.90, '', transform=ax.transAxes, color="white", fontsize=5, va='top')
text_mu = ax.text(0.05, 0.85, '', transform=ax.transAxes, color="white", fontsize=5, va='top')

kappa_init = lens_model.kappa([sra_init], [sdec_init], kwargs=kwargs_lens)[0]
g_1, g_2 = lens_model.gamma([sra_init], [sdec_init], kwargs=kwargs_lens)
g_init = (g_1[0] ** 2 + g_2[0] ** 2)**0.5
mu_init = 1 / ((1 - kappa_init)**2 - g_init**2)
    
text_kappa.set_text(f"κ: {kappa_init:.4f}")
text_gamma.set_text(f"γ: {gamma_init:.4f}")
text_mu.set_text(f"μ: {mu_init:.4f}")'''

fig = plt.figure(figsize=(15, 6))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])
plt.subplots_adjust(left=0.25, bottom=0.6)

# Gigalens
image_display = ax1.imshow(sqrt_rescale(img, scale_min=0, scale_max=0.1), origin='lower', extent=[-5, 5, -5, 5])
ax1.set_title('Gigalens')
lens_plot.caustics_plot(ax1, coords, lens_model, kwargs_lens, fast_caustic=True, color_crit='red', color_caustic='green', alpha=0.6)
ax1.plot(sra_init, sdec_init, 'bx')
temp_solver = LensEquationSolver(lens_model)
image_positions = temp_solver.image_position_from_source(sra_init, sdec_init, kwargs_lens)
mu_init = np.sum(np.abs(lens_model.magnification(*image_positions, kwargs=kwargs_lens)))
text_mu = ax1.text(0.05, 0.85, '', transform=ax1.transAxes, color="white", fontsize=5, va='top')
text_mu.set_text(f"μ: {mu_init:.4f}")

# lenstronomy
lenstronomy_display = ax2.imshow(lensstron_img, origin='lower', extent=[-5, 5, -5, 5])
lens_plot.caustics_plot(ax2, coords, lens_model, kwargs_lens, fast_caustic=True, color_crit='red', color_caustic='green', alpha=0.6)
ax2.plot(sra_init, sdec_init, 'bx')
ax2.set_title('Lenstronomy')

positive_mask = np.ma.masked_where(mag <= 0, mag) 
negative_mask = np.ma.masked_where(mag > 0, mag)

# magnification display
pos_display = ax3.imshow(
    np.log10(np.abs(positive_mask)),
    origin='lower',
    extent=[-5, 5, -5, 5],
    cmap='Reds', 
    vmin=-3.0,
    vmax= 3.0
)
neg_display = ax3.imshow(
    np.log10(np.abs(negative_mask)), 
    origin='lower',
    extent=[-5, 5, -5, 5],
    cmap='Blues',
    vmin=-3.0,
    vmax=3.0
)
'''
contour = ax2.contour(
    mag,
    levels=[0],  # Contour where magnification crosses zero
    colors='black',
    linewidths=0.5,
    extent=[-5, 5, -5, 5]
)
'''
ax3.set_title('Log of Magnification')
fig.colorbar(pos_display, ax=ax3, label='Log |Mag| (+ = red, - = blue)')
fig.colorbar(neg_display, ax=ax3)

# problem solution
ax1.set_xlim(-5,5)
ax1.set_ylim(-5,5)
ax2.set_xlim(-5,5)
ax2.set_ylim(-5,5)
ax1.set_xlabel('RA [arcsec]')
ax2.set_xlabel('RA [arcsec]')
ax3.set_xlabel('RA [arcsec]')
ax1.set_ylabel('DEC [arcsec]')

# slider setup
ax_theta_E = plt.axes([0.25, 0.4, 0.65, 0.03])
ax_q = plt.axes([0.25, 0.35, 0.65, 0.03])
ax_phi = plt.axes([0.25, 0.3, 0.65, 0.03])
ax_gamma = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_gamma_ext = plt.axes([0.25, 0.2, 0.65, 0.03])
ax_psi_ext = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_sra = plt.axes([0.25, 0.1, 0.65, 0.03])
ax_sdec = plt.axes([0.25, 0.05, 0.65, 0.03])

slider_theta_E = Slider(ax_theta_E, 'Einstein Radius (θ_E)', 0.5, 3.0, valinit=theta_E_init)
slider_q = Slider(ax_q, 'Axis Ratio (q)', 0.5, 1.0, valinit=q_init)
slider_phi = Slider(ax_phi, 'Lens Angle (φ)', -np.pi, np.pi, valinit=phi_init)
slider_gamma = Slider(ax_gamma, 'Gamma', 1.5, 3.0, valinit=gamma_init)
slider_gamma_ext = Slider(ax_gamma_ext, 'Shear Strength (γext)', 0.0, 0.3, valinit=gamma_ext_init)
slider_psi_ext = Slider(ax_psi_ext, 'Shear Angle (ψ)', -np.pi, np.pi, valinit=psi_ext_init)
slider_sra = Slider(ax_sra, 'Source RA', -2.0, 2.0, valinit=sra_init)
slider_sdec = Slider(ax_sdec, 'Source Dec', -2.0, 2.0, valinit=sdec_init)

# update function
def update(val):
    theta_E = slider_theta_E.val
    #e1 = slider_e1.val
    #e2 = slider_e2.val
    q = slider_q.val
    phi = slider_phi.val
    sra = slider_sra.val
    sdec = slider_sdec.val
    gamma = slider_gamma.val
    #gamma_1 = slider_gamma_1.val
    #gamma_2 = slider_gamma_2.val
    gamma_ext = slider_gamma_ext.val
    psi_ext = slider_psi_ext.val
    new_img, new_kwargs_lens, new_lens_model, new_coords, new_mag, lensstron_img_new = image_sim(
        theta_E, q, phi, lens_z, lra, ldec, source_z, sra, sdec, lens_light, gamma, gamma_ext, psi_ext, source_light, ps_light
    )
    
    # update text
    '''

    kappa = new_lens_model.kappa([sra], [sdec], kwargs=new_kwargs_lens)[0]
    g_1, g_2 = new_lens_model.gamma([sra], [sdec], kwargs=new_kwargs_lens)
    g = (g_1[0] ** 2 + g_2[0] ** 2)**0.5
    mu = 1 / ((1 - kappa)**2 - g**2)
    '''

    # update image
    '''
    image_display.set_data(new_img)
    ax.clear()
    ax.imshow(new_img, aspect='equal', origin='lower', extent=[-5, 5, -5, 5])
    lens_plot.caustics_plot(ax, new_coords, new_lens_model, new_kwargs_lens, fast_caustic=True, color_crit='red', color_caustic='green', alpha=0.6)
    ax.plot(sra, sdec, 'bx')
    '''
    ax1.clear()
    ax2.clear()
    ax3.clear()
    # gpt ideas
    # image_display = ax1.imshow(sqrt_rescale(new_img, scale_min=0, scale_max=0.1), origin='lower', extent=[-5, 5, -5, 5])
    image_display.set_data(new_img)
    ax1.imshow(sqrt_rescale(new_img, scale_min=0, scale_max=0.1), aspect='equal', origin='lower', extent=[-5, 5, -5, 5])
    ax1.set_title('Gigalens')
    lenstronomy_display.set_data(lensstron_img_new)
    ax2.set_title('Lenstronomy')
    ax2.imshow(lensstron_img_new, origin='lower', extent=[-5, 5, -5, 5])
    lens_plot.caustics_plot(ax1, new_coords, new_lens_model, new_kwargs_lens, fast_caustic=True, color_crit='red', color_caustic='green', alpha=0.6)
    lens_plot.caustics_plot(ax2, new_coords, new_lens_model, new_kwargs_lens, fast_caustic=True, color_crit='red', color_caustic='green', alpha=0.6)
    
    pos_display.set_data(new_mag)
    #ax2.imshow(new_mag, origin='lower', extent=[-5, 5, -5, 5], cmap='bwr')
    positive_mask = np.ma.masked_where(new_mag <= 0, new_mag)
    negative_mask = np.ma.masked_where(new_mag > 0, new_mag)
    ax3.set_title('Log of Magnification')
    ax3.imshow(
        np.log10(np.abs(positive_mask)),
        origin='lower',
        extent=[-5, 5, -5, 5],
        cmap='Reds',
        vmin=-3.0,
        vmax=3.0
    )

    ax3.imshow(
        np.log10(np.abs(negative_mask)), 
        origin='lower',
        extent=[-5, 5, -5, 5],
        cmap='Blues',
        vmin=-3.0,
        vmax=3.0
    )
    ax1.plot(sra, sdec, 'bx')
    ax2.plot(sra, sdec, 'bx')

    # problem solution
    ax1.set_xlim(-5,5)
    ax1.set_ylim(-5,5)
    ax2.set_xlim(-5,5)
    ax2.set_ylim(-5,5)
    ax1.set_xlabel('RA [arcsec]')
    ax2.set_xlabel('RA [arcsec]')
    ax3.set_xlabel('RA [arcsec]')
    ax1.set_ylabel('DEC [arcsec]')

    new_solver = LensEquationSolver(new_lens_model)
    image_positions_new = new_solver.image_position_from_source(sra, sdec, new_kwargs_lens)
    mu = np.sum(np.abs(new_lens_model.magnification(*image_positions_new, kwargs=new_kwargs_lens)))
    text_mu = ax1.text(0.05, 0.85, '', transform=ax1.transAxes, color="white", fontsize=5, va='top')
    text_mu.set_text(f"μ: {mu:.4f}")

    '''
    text_kappa = ax.text(0.05, 0.95, '', transform=ax.transAxes, color="white", fontsize=5, va='top')
    text_gamma = ax.text(0.05, 0.90, '', transform=ax.transAxes, color="white", fontsize=5, va='top')
    text_mu = ax.text(0.05, 0.85, '', transform=ax.transAxes, color="white", fontsize=5, va='top')

    text_kappa.set_text(f"κ: {kappa:.4f}")
    text_gamma.set_text(f"γ: {g:.4f}")
    text_mu.set_text(f"μ: {mu:.4f}")'''
    fig.canvas.draw_idle()

# link sliders to update function
slider_theta_E.on_changed(update)
slider_q.on_changed(update)
slider_phi.on_changed(update)
slider_sra.on_changed(update)
slider_sdec.on_changed(update)
slider_gamma.on_changed(update)
slider_gamma_ext.on_changed(update)
slider_psi_ext.on_changed(update)

plt.show()

