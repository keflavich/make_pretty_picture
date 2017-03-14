from pylab import *
test=False
if test: prefix="test_"
else: prefix=""
import AG_fft_tools
import time
t0 = time.time()

import os
os.chdir('/Volumes/disk4/w51/')

if not 'b1' in locals():
    import pyfits
    #b1 = pyfits.open('WISE_b1_W51_mosaic_bgmatch.fits')
    b1 = pyfits.open('W51_irac1.fits')
    #b2 = pyfits.open('WISE_B2_W51_mosaic_bgmatch.fits')
    #b3 = pyfits.open('WISE_b3_W51_mosaic_bgmatch.fits')
    b3 = pyfits.open('W51_irac4.fits')
    #b4 = pyfits.open('WISE_b4_W51_mosaic_bgmatch.fits')
    b4 = pyfits.open('MIPS_b1_W51_mosaic_bgmatch_fixed.fits')
    v2 = pyfits.open('W51_MOSAIC_med.fits')
    v2noise = pyfits.open('W51_MOSAIC_NOISE.fits')
    #for x in (b1,b3,b4,v2,v2noise):

    #v2noise[0].data[1082:1387,5176:5481] = 0.01


import numpy as np
import copy
import matplotlib

display_cutoff = 0.05
sncut = 2.0
mid_cut = 3.0

# Follow the ds9 definition: y = log(ax+1)/log(a) 
# or do this: 
b1min = 1.5 ; b1max=300; b1scale=3.0
b2min = 9 ; b2max=450; b2scale=3.0
b3min = 9; b3max=5600; b3scale=5.0 #1.25
b4min = 10; b4max=70000; b4scale=5.0 #0.86
def linearize(x, xmin=None, xmax=None, offset=0.0):
    if np.isscalar(x):
        return x
    else:
        if xmin is None:
            xmin = np.nanmin(x)
        if xmax is None:
            xmax = np.nanmax(x)
        return ((x-xmin)/(xmax-xmin)) / (1.0+offset) + offset
v2scale = lambda x: (np.sinh( (linearize(x)-0.1)*4.) /np.sinh(4)+1)/2.1
v2scale = lambda x: np.log10(1000*linearize(x)+1)/np.log10(1000)
v2loscale = lambda x: (np.arcsinh(linearize(x)*10)/np.arcsinh(10))**0.5 #lambda x: x  # was np.log10
v2loscale = lambda x: linearize(np.tan(linearize(x)-0.55)**3)
v2loscale = lambda x: linearize( np.log10(1000*linearize(x)+1)/np.log10(1000), offset=0.2)
v2scale = lambda x: (np.arcsinh(linearize(x)*10)/np.arcsinh(10)) #lambda x: x  # was np.log10
def logscale(array, logexp=3.0, **kwargs):
    linarr = linearize(arr, **kwargs)
    return np.log10(linarr * 10**logexp + 1)


if not 'v2scaled' in locals():
    if test:
        myslice = slice(None,None,4),slice(None,None,4) 
        myslice = slice(2048,3072),slice(0,1024)#slice(3072,4096)#1024,2048)#2048,3072)
        rgb = np.ones([1024,1024,4])
        rgb[:,:,0] = (np.log10((b4[0].data[myslice]-b4min)/b4max*10**b4scale+1))/b4scale; rgb[:,:,0][rgb[:,:,0]<0] = 0; rgb[:,:,0][rgb[:,:,0]>1] = 1
        rgb[:,:,1] = (np.log10((b3[0].data[myslice]-b3min)/b3max*10**b3scale+1))/b3scale; rgb[:,:,1][rgb[:,:,1]<0] = 0; rgb[:,:,1][rgb[:,:,1]>1] = 1
        #rgb[:,:,2] = (np.log10((b2[0].data[myslice]-b2min)/b2max*10**b2scale+1))/b2scale; rgb[:,:,2][rgb[:,:,2]<0] = 0; rgb[:,:,2][rgb[:,:,2]>1] = 1
        rgb[:,:,2] = (np.log10((b1[0].data[myslice]-b1min)/b1max*10**b1scale+1))/b1scale; rgb[:,:,2][rgb[:,:,2]<0] = 0; rgb[:,:,2][rgb[:,:,2]>1] = 1
        #v2scaled = np.log10(AG_fft_tools.smooth(v2[0].data[myslice]))
        v2d = v2[0].data[myslice]
        v2d[v2d!=v2d] = 0
        v2smooth = np.fft.fftshift( np.fft.ifft2( np.fft.fft2(v2d*(v2d>display_cutoff)) * np.fft.fft2(AG_fft_tools.smooth_tools.make_kernel([1024,1024])) ) ).real
        v2scaled = v2scale(v2smooth)#AG_fft_tools.smooth(v2[0].data*(v2[0].data>0.025)))
        v2scaled[v2scaled!=v2scaled] = 0
        v2scaled[np.isinf(v2scaled)] = 0
        v2rescaled=v2resmooth=v2scaled
        v2mid = v2loscale(v2smooth*(v2smooth<mid_cut))
        nanhigh = np.ones(v2smooth.shape)
        nanhigh[(v2smooth<mid_cut)] = np.nan
        v2high = v2scale(v2smooth*nanhigh)
        v2high[np.isnan(v2high)] = 0

        #v2rescaled = v2scale(v2smooth*(v2smooth>display_cutoff)) #(v2scale(v2smooth*(v2smooth>display_cutoff))-v2scale(display_cutoff))/(np.nanmax(v2scaled)-v2scale(display_cutoff))
        #v2rescaled[True - np.isfinite(v2rescaled)] = 0
        #v2resmooth = np.fft.fftshift( np.fft.ifft2( np.fft.fft2(v2d*(v2d>display_cutoff)) * np.fft.fft2(AG_fft_tools.smooth_tools.make_kernel(v2rescaled.shape,5)) ) ).real
        #v2resmooth[v2scaled>v2scale(2*display_cutoff)] = v2rescaled[v2scaled>v2scale(2.*display_cutoff)]
    else:
        #myshape = (2700,7200)
        myshape=v2[0].data.shape
        myslice = slice(None,None,None),slice(None,None,None)
        rgb = np.ones([myshape[0],myshape[1],4])
        rgb[:,:,0] = (np.log10((b4[0].data[myslice]-b4min)/b4max*10**b4scale+1))/b4scale; rgb[:,:,0][rgb[:,:,0]<0] = 0; rgb[:,:,0][rgb[:,:,0]>1] = 1
        rgb[:,:,1] = (np.log10((b3[0].data[myslice]-b3min)/b3max*10**b3scale+1))/b3scale; rgb[:,:,1][rgb[:,:,1]<0] = 0; rgb[:,:,1][rgb[:,:,1]>1] = 1
        #rgb[:,:,2] = (np.log10((b2[0].data[myslice]-b2min)/b2max*10**b2scale+1))/b2scale; rgb[:,:,2][rgb[:,:,2]<0] = 0; rgb[:,:,2][rgb[:,:,2]>1] = 1
        rgb[:,:,2] = (np.log10((b1[0].data[myslice]-b1min)/b1max*10**b1scale+1))/b1scale; rgb[:,:,2][rgb[:,:,2]<0] = 0; rgb[:,:,2][rgb[:,:,2]>1] = 1
        v2d = v2[0].data[myslice]
        v2n = v2noise[0].data[myslice]
        v2d[v2d!=v2d] = 0
        v2d[v2d/v2n < sncut] = 0
        v2smooth = np.fft.fftshift( np.fft.ifft2( np.fft.fft2(v2d*(v2d>display_cutoff)) * np.fft.fft2(AG_fft_tools.smooth_tools.make_kernel([myshape[0],myshape[1]])) ) ).real
        v2scaled = v2scale(v2smooth)#AG_fft_tools.smooth(v2[0].data*(v2[0].data>0.025)))
        v2scaled[np.isinf(v2scaled) + (v2scaled!=v2scaled)] = 0
        v2rescaled=v2resmooth=v2scaled

        v2mid = v2scale(v2smooth*(v2smooth<mid_cut))
        nanhigh = np.ones(v2smooth.shape)
        nanhigh[(v2smooth<mid_cut)] = np.nan
        v2high = v2scale(v2smooth*nanhigh)
        v2high[np.isnan(v2high)] = 0

        #v2bad = v2d == 0
        #v2badmask = np.fft.fftshift( np.fft.ifft2( np.fft.fft2(v2bad.astype('float')) * np.fft.fft2(AG_fft_tools.smooth_tools.make_kernel([myshape[0],myshape[1]],30,kerneltype='tophat')) ) ).real
        #v2resmooth[v2badmask>0.1] = 0
        #v2mid[v2badmask>0.1] = 0
        #v2high[v2badmask>0.1] = 0
        #v2rescaled = (v2scale(v2smooth*(v2smooth>display_cutoff))-v2scale(display_cutoff))/(np.nanmax(v2scaled)-v2scale(display_cutoff))
        #v2rescaled[True - np.isfinite(v2rescaled)] = 0
        #v2resmooth = np.fft.fftshift( np.fft.ifft2( np.fft.fft2(v2d*(v2d>display_cutoff)) * np.fft.fft2(AG_fft_tools.smooth_tools.make_kernel(v2rescaled.shape,5)) ) ).real
        #v2resmooth[v2scaled>v2scale(2*display_cutoff)] = v2rescaled[v2scaled>v2scale(2.*display_cutoff)]
    rgb[rgb!=rgb]=0

print "Creating scaling ",time.time()-t0
#v2scaled[v2scaled < np.log10(0.025)] = np.nan
autumn_transparent = copy.copy(matplotlib.cm.autumn)
#autumn_transparent._lut = matplotlib.cm.autumn(np.linspace(0.4,1.0,autumn_transparent.N+3))
autumn_transparent._lut = matplotlib.cm.Oranges_r(np.linspace(0.2,0.8,autumn_transparent.N+3))
autumn_transparent._lut[:,1] *= 1.5
autumn_transparent._lut[autumn_transparent._lut[:,1]>1,1] = 1
autumn_transparent._isinit=True
#autumn_transparent._lut = matplotlib.cm.autumn_r(np.linspace(0.0,0.7,autumn_transparent.N+3))
#autumn_transparent._lut[:,2] *= 0.1
autumn_transparent._lut[:,3] *= np.sin(np.linspace(0.1,0.8,autumn_transparent.N+3)*pi/2)**2
autumn_transparent._lut[256,:] = 0
autumn_transparent._lut[257,:] = autumn_transparent._lut[255,:]
autumn_transparent._lut[258,:] = 0
#autumn_transparent._lut *= np.outer(np.linspace(0.1,1,autumn_transparent.N),np.ones(4))

high_lut = {
        'red':[(0,autumn_transparent(1-1e-5)[0],autumn_transparent(1-1e-5)[0]),(1,1,1)],
        'green':[(0,autumn_transparent(1-1e-5)[1],autumn_transparent(1-1e-5)[1]),(1,0.5,0.5)],
        'blue':[(0,autumn_transparent(1-1e-5)[2],autumn_transparent(1-1e-5)[2]),(1,0,0)],
        }
high = matplotlib.colors.LinearSegmentedColormap('high',high_lut)
high._lut = high(np.linspace(0,1,high.N))
high._lut[:,3] = np.linspace(0.8,1.0,high.N)

rgb[:,:,3] = rgb[:,:,:3].mean(axis=2)


print "Creating scaled image ",time.time()-t0
v2img = autumn_transparent(v2resmooth)
v2img = autumn_transparent(v2mid)*(v2mid>1e-6)[:,:,newaxis]+high(v2high)*((v2mid<=1e-6)*(v2high>1e-6))[:,:,newaxis]
#v2img = autumn_transparent((v2scaled-np.log10(display_cutoff))/(np.nanmax(v2scaled)-np.log10(display_cutoff)))
#print autumn_transparent._lut[:,3], autumn_transparent._isinit
if v2img.sum() == float(v2img.shape[0])**2: raise
#v2img = autumn_transparent((v2scaled-np.nanmin(v2scaled))/(np.nanmax(v2scaled)-np.nanmin(v2scaled)))
#v2img[:,:,:3] = v2img[:,:,:3]*(v2img[:,:,3][:,:,np.newaxis])**0.25
rgb2 = rgb.copy()
# reduce blueness...
# rgb2[:,:,2] -= v2img[:,:,:2].mean(axis=2)
#rgb2[v2scaled>log10(display_cutoff),:] += v2img[v2scaled>log10(display_cutoff),:] 
#rgb2[v2scaled>log10(display_cutoff),:3] *= np.min( np.concatenate([[v2img[:,:,:3]/ (v2img[:,:,3][:,:,np.newaxis])**0.5], [np.ones(v2img.shape[:2]+(3,))]]), axis=0 )[v2scaled>log10(display_cutoff)] 
#rgb2[v2scaled>log10(display_cutoff),3] += (v2img[v2scaled>log10(display_cutoff),3])**2.0
colorscale = np.nanmin( np.concatenate([[v2img[:,:,:3]/ (v2img[:,:,3][:,:,newaxis])**1.0], [np.ones(v2img.shape[:2]+(3,))]]), axis=0 )
rgb2[:,:,:2] *= colorscale[:,:,:2]
rgb2[:,:,2] *= colorscale[:,:,2] + colorscale[:,:,2]==0
rgb2[:,:,3] += (v2img[:,:,3])**2.0
rgb2[rgb2>1] = 1

figure(2)
clf()
ax1=subplot(111,axisbg='k')
imshow(rgb)
print "Saving RGB mosaic ",time.time()-t0
savefig(prefix+"W51_WISE_mosaic.png",dpi=300)
imshow(v2img)
print "Saving RGB +bolo mosaic ",time.time()-t0
savefig(prefix+"W51_WISE_mosaic_bolocam.png",dpi=300)
draw()

fig=figure(1)
clf()
ax=subplot(111,axisbg='k')
imshow(rgb2)
print "Saving RGB+bolo try 2 ",time.time()-t0
savefig(prefix+"W51_WISE_mosaic_bolocam_try2.png",dpi=300)
#imshow(v2scaled[2048:3084,2048:3084], cmap=autumn_transparent)
draw()

figure(3)
clf()
#imshow(v2scaled, cmap=autumn_transparent)
ax=subplot(111,axisbg='k')
imshow(v2img)#, cmap=autumn_transparent)
print "Saving Bolocam img ",time.time()-t0
savefig(prefix+'W51_bolocam_orange.png',dpi=300)
colorbar()
draw()

figure(4)
clf()
imshow(v2img[:,:,3])#, cmap=autumn_transparent)
colorbar()
draw()

figure(5)
clf()
ax=subplot(111,axisbg='k')
imshow(colorscale.sum(axis=2))
colorbar()
draw()

figure(6)
clf()
imshow(v2scaled)
colorbar()
draw()

figure(7)
clf()
imshow(v2resmooth)
colorbar()
draw()

if test:
    pilslice = slice(None,None,None)
else:
    pilslice = slice(None,None,None)

print "Beginning PIL operations ",time.time()-t0
import PIL,ImageEnhance,ImageOps
rgb_pil = ((rgb)*(255)).astype('uint8')[pilslice,:,:]
rgb_pil = rgb_pil[::-1,:,:]
rgb_pil[np.max(rgb_pil,axis=2)>=255,:] = 255
#rgb_pil[:,:,3] = np.uint8(256)-rgb_pil[:,:,3]
im1 = PIL.Image.fromarray(rgb_pil)
print "Saving WISE mosaic ",time.time()-t0
im1.save(prefix+'W51_4096sq_WISE_mosaic.png')
#rgb_pil = ((1-rgb[:,:,:3])*(2**8))
#rgb_pil -= (256*rgb[:,:,3])[:,:,newaxis]
#rgb_pil[rgb_pil>255] = 255
#rgb_pil = rgb_pil.astype('uint8')
#im1 = PIL.Image.fromarray(rgb_pil)
#im1.save(prefix+'W51_4096sq_WISE_mosaic_try2.png')
print "Saving WISE mosaic with white bg ",time.time()-t0
wbackground = PIL.Image.new("RGB", im1.size, (255, 255, 255))
wbackground.paste(im1, mask=im1.split()[3])
wbackground.save(prefix+'W51_4096sq_WISE_mosaic_whitebg.png')
kbackground = PIL.Image.new("RGB", im1.size, (0, 0, 0))
kbackground.paste(im1, mask=im1.split()[3])
print "Saving WISE mosaic with black bg ",time.time()-t0
kbackground.save(prefix+'W51_4096sq_WISE_mosaic_blackbg.png')
kbackground_contrast = ImageOps.autocontrast(kbackground)
kbackground_contrast.save(prefix+'W51_4096sq_WISE_mosaic_blackbg_contrast.png')
kbackground_bright = ImageEnhance.Brightness(kbackground_contrast).enhance(1.5)
kbackground_bright.save(prefix+'W51_4096sq_WISE_mosaic_blackbg_contrast_bright.png')

print "doing Bolocam PIL stuff ",time.time()-t0
v2pil = (v2img*255).astype('uint8')[pilslice,:,:]
v2pil = v2pil[::-1,:,:]
boloim = PIL.Image.fromarray(v2pil)
boloim.save(prefix+'W51_4096sq_bolo.png')
kbackground.paste(boloim, mask=boloim.split()[3])
print "Saving Bolocam + WISE mosaic with black bg (PIL) ",time.time()-t0
kbackground.save(prefix+'W51_4096sq_WISE_bolo_mosaic_blackbg.png')

print "Done ",time.time()-t0

rotated = rgb_pil.copy()
rotated[:,:,0] = rgb_pil[:,:,2]
rotated[:,:,1] = rgb_pil[:,:,0]
rotated[:,:,2] = rgb_pil[:,:,1]
im2 = PIL.Image.fromarray(rotated)
print "Saving WISE mosaic ",time.time()-t0
im2.save(prefix+'W51_4096sq_WISE_mosaic_rotated.png')
kbackground2 = PIL.Image.new("RGB", im2.size, (0, 0, 0))
kbackground2.paste(im2, mask=im2.split()[3])
print "Saving WISE mosaic with black bg ",time.time()-t0
kbackground2.save(prefix+'W51_4096sq_WISE_mosaic_rotated_blackbg.png')
kbackground2_small = kbackground2.resize([s/4 for s in kbackground2.size])
kbackground2_small.save(prefix+'W51_4096sq_WISE_mosaic_rotated_blackbg_small.png',quality=10)
print "Saving Bolocam + WISE mosaic with black bg (PIL) ",time.time()-t0
kbackground2_contrast = ImageOps.autocontrast(kbackground2)
kbackground2_contrast.save(prefix+'W51_4096sq_WISE_mosaic_rotated_blackbg_contrast.png')
kbackground2_bright = ImageEnhance.Brightness(kbackground2_contrast).enhance(1.5)
kbackground2_bright.save(prefix+'W51_4096sq_WISE_mosaic_rotated_blackbg_contrast_bright.png')
kbackground2_small_bright = kbackground2_bright.resize([s/4 for s in kbackground2.size])
kbackground2_small_bright.save(prefix+'W51_4096sq_WISE_mosaic_rotated_blackbg_bright_small.png',quality=10)
kbackground2_bright.paste(boloim, mask=boloim.split()[3])
kbackground2_bright.save(prefix+'W51_4096sq_WISE_bolo_mosaic_rotated_blackbg.png')
kbackground2_small = kbackground2_bright.resize([s/4 for s in kbackground2.size])
kbackground2_small.save(prefix+'W51_4096sq_WISE_bolo_mosaic_rotated_blackbg_small.png',quality=10)
print "Done rotated ",time.time()-t0
