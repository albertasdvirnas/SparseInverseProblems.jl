cd("git/SparseInverseProblems.jl")
# Function to test running 2d simple ADCG


# import Pkg;

# this uses a forked version
# git clone https://github.com/marius311/SparseInverseProblems.jl.git
Pkg.clone("git@github.com:albertasdvirnas/SparseInverseProblems.jl.git")
# which includes a toml file and some
# modifications for this to run in Julia properly

# load all necessary functions.
using Pkg
Pkg.activate(".")
using SparseInverseProblems
using Compose
using SpecialFunctions
using Images
using Plots
import SparseInverseProblems: lmo, phi, solveFiniteDimProblem, localDescent,
getStartingPoint, parameterBounds, computeGradient
using SparseInverseProblems.Util
using SparseInverseProblems
using NLopt, SparseInverseProblems.Util
using LinearAlgebra
include("examples/smi/gaussblur.jl")

# functions which are changed from default..

function computeGradient(model :: GaussBlur2D, weights :: Vector{Float64},
  thetas :: Matrix{Float64}, v :: Vector{Float64})
  gradient=0
  v = reshape(v, model.n_pixels, model.n_pixels)
  #
  gradient = zeros(length(thetas))
  # #allocate temporary variables...
  f_x = zeros(model.n_pixels)
  f_y = zeros(model.n_pixels)
  fpy = zeros(model.n_pixels)
  fpx = zeros(model.n_pixels)
  v_x = zeros(model.n_pixels)
  v_y = zeros(model.n_pixels)
  v_yp = zeros(model.n_pixels)

  # #compute gradient
  for i = 1:size(thetas,2)
    point = vec(thetas[:,i])
    computeFG(model, point[1], f_x, fpx)
    computeFG(model, point[2], f_y, fpy)
    v_x = mul!(v_x,v,f_x)
    v_y = mul!(v_y,v,f_y)
    v_yp = mul!(v_yp,v,fpy)
    function_value = dot(f_y, v_x)
    g_x = dot(v_y, fpx)
    g_y = dot(v_x, fpy)

    gradient[:,i] = weights[i]*[g_x; g_y]
  end
  return gradient
end

function gensim()

  noise_mean = 0.0021172377176794793
  sigmasq = let
    lambda = 723.0
    NA = 1.4
    FWHM = lambda/(2*NA)
    sigma = FWHM/(2*log(2.0))
    (sigma/(64*100.0))^2
  end


  gb_sim = GaussBlur2D(sigmasq,64,500)


  return gb_sim
end


function phi(s :: GaussBlur2D, parameters :: Matrix{Float64},weights :: Matrix{Float64})
  n_pixels = s.n_pixels
  if size(parameters,2) == 0
    return zeros(n_pixels*n_pixels)
  end
  v_x = computeFs(vec(parameters[1,:]),n_pixels,s.sigma).*weights
  v_y = computeFs(vec(parameters[2,:]),n_pixels,s.sigma)
  return vec(v_y*v_x')
end


# Generate simulation
gb_sim = gensim()

parameters = rand(2,4);
weights = 100000*rand(1,4);

# this gives us the output
y = phi(gb_sim,parameters, weights)

# for showing heatmap, we need to reshape to 64x64 (or change how these are chosen in gb_sim)
heatmap(reshape(y,64,64))

# callback function. remove mean  noise from target
target = vec(y).-noise_mean;
function callback(old_thetas,thetas, weights,output,old_obj_val)
  #evalute current OV
  new_obj_val,t = loss(LSLoss(), output - target)
  if old_obj_val - new_obj_val < 7E-5
    return true
  end
  return false
end

# now default parameters for ADCG calculation
lossFn =LSLoss()
tau = 100.0;
max_iters = 50
min_optimality_gap = 1E-5
max_cd_iters = 200
fully_corrective = false
@assert(tau > 0.0)

output = zeros(length(y))

residual = output - y
#evalute the objective value and gradient of the loss
objective_value, grad = loss(lossFn, residual)
#compute the next parameter value to add to the support

# end
# next parameter value and score. Using the default gives us
# score = 0.0..
theta,score = lmo(gb_sim, grad)

# go through lmo methods step by step.
# 1 starting point
initial_x = getStartingPoint(gb_sim, grad)
#
# v = reshape(grad, gb_sim.n_pixels, gb_sim.n_pixels)
#
# # grad2 = reshape(grad, gb_sim.n_pixels, gb_sim.n_pixels)
# ng = length(gb_sim.grid)
# grid_objective_values = vec(gb_sim.grid_f'*v*gb_sim.grid_f)
# best_point_idx = argmin(grid_objective_values)

# parameters
model = gb_sim
v = residual

lb,ub = parameterBounds(model)
initial_x = getStartingPoint(model, v)
println(initial_x)
p = length(lb)

function fg!(point :: Vector{Float64},gradient_storage :: Vector{Float64})
  output = phi(model,reshape(point,p,1), ones(1,1))
  ip = dot(output,v)
  s = sign(ip)
  gradient_storage[:] = -s*computeGradient(model, [1.0],reshape(point,length(point),1), v)
  # gradient_storage[:] = -s*computeGradient(model, [1.0],reshape(point,length(point),1), v)
  return -s*ip
end

 phi(model,reshape(initial_x,2,1),ones(1,1))

 function initializeOptimizer!(model :: BoxConstrainedDifferentiableModel, opt :: Opt)
  ftol_abs!(opt, 1e-6)
  xtol_rel!(opt, 0.0)
  maxeval!(opt, 200)
end


#
# i=1
# dpsi(model, vec(initial_x[:,i]))'*v
# computeGradient(model,ones(1,1))

opt = Opt(:LD_MMA, p)
initializeOptimizer!(model, opt)
min_objective!(opt, fg!)
# opt.min_objective = f_and_g!
lower_bounds!(opt, lb)
upper_bounds!(opt, ub)
(optf,optx,ret) = optimize(opt, initial_x)

# Niw optf is non-zero.
println(optf)
println(optx)

#
#
# weights = vec(ones(1,1))
# thetas = reshape(initial_x,2,1)
# v = residual
# grfff = computeGradient(model, weights,thetas, v)
