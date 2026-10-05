"""Sample the saved polynomial or neural Lyapunov function without clipping."""
import numpy as np
from .polynomial import features


def sample_lyapunov(model, bounds, resolution=121):
    bounds=np.asarray(bounds)
    xs=np.linspace(bounds[0,0],bounds[1,0],resolution)
    ys=np.linspace(bounds[0,1],bounds[1,1],resolution)
    xx,yy=np.meshgrid(xs,ys)
    points=np.column_stack([xx.ravel(),yy.ravel()])
    neural=model.get('config',{}).get('method')=='snds'
    values=(model['policy'].potential(points) if neural else
            features(points,model['lyapunov_basis']) @ model['vcoeff']).reshape(resolution,resolution)
    if not np.isfinite(values).all():raise ValueError('Nonfinite Lyapunov values')
    # Signed asinh keeps zero and negative numerical values visible, while
    # resolving the target region when a high-degree polynomial grows quickly.
    scale=max(float(np.percentile(np.abs(values),50)),1e-12)
    transformed=np.arcsinh(values/scale)
    low=min(float(transformed.min()),0.0)
    high=max(float(transformed.max()),low+1e-12)
    ticks=scale*np.sinh(np.linspace(low,high,5))
    return dict(bounds=bounds.tolist(),resolution=resolution,values=values.tolist(),
                scale=scale,transformed_range=[low,high],ticks=ticks.tolist(),
                minimum=float(values.min()),maximum=float(values.max()),
                normalization='asinh',source='saved neural potential' if neural else 'saved polynomial coefficients')
