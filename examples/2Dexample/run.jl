using SparseInverseProblems
cd("examples/2Dexample/")
include("gauss2d.jl")

# Download data.
if !isdir("data")
  print("Downloading data...")
  run(`./get-data.sh`)
  run(`mkdir output`)
  println("done.")
end


function gensim()
  noise_mean = 0.0021172377176794793
  sigmasq = let
    lambda = 723.0
    NA = 1.4 # numerical aparature
    FWHM = lambda/(2*NA) # full width half maximum
    sigma = FWHM/(2*log(2.0))
    (sigma/(64*100.0))^2
  end


  gb_sim = GaussBlur2D(sigmasq,64,500)

  parameters = rand(2,4);
  weights = 100000*rand(1,4);

  # this gives us the output
  y = phi(gb_sim,parameters, weights)


  return gb_sim, y, parameters, weights,noise_mean
end



function runFW(sim,imageArray; n_cd_iters :: Int64 = 200, noise_mean :: Float64 = 0.2)
  nImages = size(imageArray,3)
  # results = Array(Array{Float64},nImages)
  for imageIdx = 1:nImages
    img = imageArray[:,:,imageIdx]
    target = vec(img).-noise_mean
    function callback(old_thetas,thetas, weights,output,old_obj_val)
      #evalute current OV
      new_obj_val,t = loss(LSLoss(), output - target)
      if old_obj_val - new_obj_val < 7E-5
        return true
      end
      return false
    end
    tau = 100.0;
    max_iters = 50
    (points,weights) = ADCG(gb_sim, LSLoss(), target, tau; max_iters, callback=callback, max_cd_iters=n_cd_iters)
    results[imageIdx] = points
    print(".")
  end
  results
end


#
# Generate simulation
gb_sim,y,parameters,weights,noise_mean = gensim()
heatmap()
imageArray = reshape(y,64,64,1)

results = runFW(gb_sim,imageArray,noise_mean)






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
# include("gauss2d.jl")
# using BoxConstrainedDifferentiableModel
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

 using NLopt, SparseInverseProblems.Util

 opt = Opt(:LD_MMA, p)

 function initializeOptimizer!(model :: BoxConstrainedDifferentiableModel, opt :: Opt)
  ftol_abs!(opt, 1e-6)
  xtol_rel!(opt, 0.0)
  maxeval!(opt, 200)
end


#
# i=1
# dpsi(model, vec(initial_x[:,i]))'*v
# computeGradient(model,ones(1,1))

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
