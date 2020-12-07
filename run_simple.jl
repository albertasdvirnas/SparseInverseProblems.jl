cd("/home/albyback/git/dotlabelproject/OuterProject/SparseInverseProblems.jl/")

# this uses a forked version
# git clone https://github.com/marius311/SparseInverseProblems.jl.git
# which includes a toml file and some
# modifications for this to run in Julia properly

using Pkg
Pkg.activate(".")
using SparseInverseProblems
using Compose
# cd("/home/albyback/git/dotlabelproject/OuterProject/SparseInverseProblems.jl/examples/simple_example/")
include("/home/albyback/git/dotlabelproject/OuterProject/SparseInverseProblems.jl/examples/simple_example/simple_example.jl")

#Feel free to override
function localUpdate(sim :: ForwardModel,lossFn :: Loss,
    thetas :: Matrix{Float64}, y :: Vector{Float64}, tau :: Float64, max_iters)
  for cd_iter = 1:max_iters
    weights = solveFiniteDimProblem(sim, lossFn, thetas, y, tau)
    #remove points with zero weight
    if any(weights.==0.0)
      println("Removing ",sum(weights.==0.0), " zero-weight points.")
      thetas = thetas[:,weights.!= 0.0]
      weights = weights[weights.!= 0.0]
    end
    #local minimization over the support
    new_thetas = localDescent(sim, lossFn, thetas,weights, y)
    #break if termination condition is met
    if length(thetas) == length(new_thetas) && maximum(abs.(vec(thetas)-vec(new_thetas))) <= 1E-7
        break
    end
    thetas = new_thetas
  end
  #final prune
  weights = solveFiniteDimProblem(sim, lossFn, thetas, y, tau)
  if any(weights.==0.0)
    println("Removing ",sum(weights.==0.0), " zero-weight points.")
    thetas = thetas[:,weights.!= 0.0]
    weights = weights[weights.!= 0.0]
  end
  return thetas, weights
end


function run_sample()
      evaluation_points = -5.0:0.25:5.0
      model = SimpleExample(evaluation_points, -10.0:0.5:10.0)
      means = [-2.0,1.0,3.5]
      k = length(means)
      weights = ones(k)+randn(k)*0.3
      weights[1] = 0.2
      target = max.(phi(model,Matrix(means'),weights) + randn(length(evaluation_points))*0.1,0.0);
      # (means_est,weights_est) = ADCG(model, LSLoss(), target, 100.0;  callback=callback)

      sim = model
      lossFn =LSLoss()
      y = target
      tau = 100.0;
      max_iters = 50
      min_optimality_gap = 1E-5
      max_cd_iters = 200
      fully_corrective = false
      @assert(tau > 0.0)

      bound = -Inf
      thetas = zeros(0,0) #hack
      weights = zeros(0)
      #cache the forward model applied to the current measure.
      output = zeros(length(y))

      iter = 1

      #compute the current residual
      residual = output - y
      #evalute the objective value and gradient of the loss
      objective_value, grad = loss(lossFn, residual)
      #compute the next parameter value to add to the support
      theta,score = lmo(sim,grad)
      print(theta)
      print(score)

      old_thetas = thetas
      thetas = iter == 1 ? reshape(theta, length(theta),1) : [thetas theta]
      #run local optimization over the support.
      old_weights = copy(weights)
      thetas,weights = localUpdate(sim,lossFn,thetas,y,tau,max_cd_iters)


      #score is - |<\psi(theta), gradient>|
      #update the lower bound on the optimal value
      # bound = max(bound, objective_value+score*tau-dot(output,grad))
      #check if the bound is met.
      # if (objective_value < bound + min_optimality_gap || score >= 0.0)
      # return thetas,weights
      # end
end

run_sample()

using Plots

plot(evaluation_points,y)


# # want to run a 1D example
# abstract BoxConstrainedDifferentiableModel <: ForwardModel
#
# #compute the measurement model
# psi(model :: BoxConstrainedDifferentiableModel, theta :: Vector{Float64})
#
# #compute the jacobian of the forward model
# dpsi(model :: BoxConstrainedDifferentiableModel, theta :: Vector{Float64})
#
# # Initial starting point for continuous optimization for the FW step.
# # Should return a good guess for $\arg\min_\theta \langle \psi(theta), v \rangle.$
# # Often computed using a grid.
# getStartingPoint(model :: BoxConstrainedDifferentiableModel, v :: Vector{Float64}) =
#   error("getStartingPoint not implemented for model $(typeof(model)).")
#
# # Box constraints on the parameters.
# # Returns a tuple of two vectors : lower bounds and upper bounds.
# parameterBounds(model :: BoxConstrainedDifferentiableModel)
