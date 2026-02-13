using RxInfer, Distributions, Plots, StableRNGs

# Set random seed for reproducibility
rng = StableRNG(42)

# Model parameters
n_states = 2
T = 256

A = [1.0 1.0; 0.0 0.9]
Q = [0.1 0.0; 0.0 0.1]
C = [1.0, 0.0]
R = 5.0
μ₀ = [0.0, 0.0]
Σ₀ = [1.0 0.0; 0.0 1.0]

# Generate synthetic data
function generate_lgssm_data(rng, A, Q, C, R, μ₀, Σ₀, T)
    x = Vector{Float64}[]
    y = Float64[]
    push!(x, rand(rng, MvNormal(μ₀, Σ₀)))
    for t in 1:T
        if t > 1
            push!(x, A * x[end] + rand(rng, MvNormal(zeros(size(A, 1)), Q)))
        end
        push!(y, dot(C, x[end]) + rand(rng, NormalMeanVariance(0, R)))
    end
    return x, y
end

true_states, observations = generate_lgssm_data(rng, A, Q, C, R, μ₀, Σ₀, T)

# --- Helper functions ---

function create_batched_observations(data, batch_size)
    batches = []
    for i in 1:batch_size:length(data)
        end_idx = min(i + batch_size - 1, length(data))
        push!(batches, data[i:end_idx])
    end
    return batches
end

# --- Models ---

@model function lgssm_batched(y, x_prev, A, Q, C, R, batch_size)
    x[1] ~ MvNormal(mean = A * x_prev, cov = Q)
    y[1] ~ Normal(mean = dot(C, x[1]), var = R)
    for t in 2:batch_size
        x[t] ~ MvNormal(mean = A * x[t-1], cov = Q)
        y[t] ~ Normal(mean = dot(C, x[t]), var = R)
    end
end

@model function lgssm_static(y, A, Q, C, R, μ₀, Σ₀)
    x₀ ~ MvNormal(mean = μ₀, cov = Σ₀)
    x[1] ~ MvNormal(mean = A * x₀, cov = Q)
    y[1] ~ Normal(mean = dot(C, x[1]), var = R)
    for t in 2:length(y)
        x[t] ~ MvNormal(mean = A * x[t-1], cov = Q)
        y[t] ~ Normal(mean = dot(C, x[t]), var = R)
    end
end

# --- Static (full) inference as baseline ---

println("Running static inference (baseline)...")
# Warmup run to compile everything
infer(
    model = lgssm_static(A = A, Q = Q, C = C, R = R, μ₀ = μ₀, Σ₀ = Σ₀),
    data = (y = observations,),
    free_energy = true
)
GC.gc()
# Timed run (post-JIT)
static_time = @elapsed static_results = infer(
    model = lgssm_static(A = A, Q = Q, C = C, R = R, μ₀ = μ₀, Σ₀ = Σ₀),
    data = (y = observations,),
    free_energy = true
)
static_fe = static_results.free_energy[end]
static_means = mean.(static_results.posteriors[:x])

println("  Static FE = $static_fe, time = $(round(static_time, digits=4))s")

# --- Sweep batch sizes (powers of 2) ---

batch_sizes = [2^k for k in 1:Int(log2(T))]  # 2, 4, 8, ..., T
results = []

for bs in batch_sizes
    n_batches = ceil(Int, T / bs)
    batched_obs = create_batched_observations(observations, bs)

    autoupdates = @autoupdates begin
        x_prev = mean(q(x[bs]))
    end

    init = @initialization begin
        q(x) = MvNormal(mean = μ₀, cov = Σ₀)
    end

    # Warmup + timing: run twice, time the second
    for run in 1:2
        GC.gc()
        t = @elapsed engine = infer(
            model = lgssm_batched(A = A, Q = Q, C = C, R = R, batch_size = bs),
            data = (y = batched_obs,),
            autoupdates = autoupdates,
            initialization = init,
            returnvars = (:x,),
            keephistory = n_batches,
            autostart = true,
            free_energy = true
        )

        if run == 2
            batched_fe = sum(engine.free_energy_final_only_history)
            merged = vcat(engine.history[:x]...)
            batched_means = mean.(merged)

            # MSE of posterior means vs static posterior means (state component 1)
            mse_vs_static = mean((getindex.(batched_means, 1) .- getindex.(static_means, 1)) .^ 2)

            push!(results, (
                batch_size = bs,
                n_batches = n_batches,
                time = t,
                batched_fe = batched_fe,
                fe_diff = batched_fe - static_fe,
                fe_ratio = batched_fe / static_fe,
                mse_vs_static = mse_vs_static,
                time_ratio = t / static_time,
                posterior_means = getindex.(batched_means, 1),
                posterior_stds = getindex.(std.(merged), 1, 1)
            ))
            println("  batch_size=$bs ($(n_batches) batches): FE=$(round(batched_fe, digits=2)), " *
                    "ΔFE=$(round(batched_fe - static_fe, digits=2)), " *
                    "MSE_vs_static=$(round(mse_vs_static, digits=6)), " *
                    "time=$(round(t, digits=4))s ($(round(t / static_time, digits=2))x static)")
        end
    end
end

fe_diffs = getfield.(results, :fe_diff)
time_ratios = getfield.(results, :time_ratio)

fe_norm = fe_diffs ./ maximum(abs.(fe_diffs))
tr_norm = time_ratios ./ maximum(time_ratios)

speed_weights = [1e5, 10.0, 1.0, 1e-5]
weight_labels = ["Strongly prefer speed (w=$(speed_weights[1]))",
                 "Prefer speed (w=$(speed_weights[2]))",
                 "Balanced (w=$(speed_weights[3]))",
                 "Strongly prefer accuracy (w=$(speed_weights[4]))"]
markers = [:circle, :square, :diamond, :utriangle]
colors = [:red, :orange, :steelblue, :blue]

subplots = []
for (w, wlabel, m, c) in zip(speed_weights, weight_labels, markers, colors)
    combined = w .* tr_norm .+ fe_norm
    best_idx = argmin(combined)

    p = plot(bs_labels, combined,
        ylabel = "Cost", xlabel = "Batch size",
        title = wlabel,
        marker = m, markersize = 5, linewidth = 2, color = c,
        legend = false)
    scatter!([bs_labels[best_idx]], [combined[best_idx]],
        marker = :star5, markersize = 12, color = :green)
    annotate!(best_idx, combined[best_idx] + 0.05 * maximum(combined),
        text("bs=$(results[best_idx].batch_size)", 8, :green))

    push!(subplots, p)
end

p_sweet = plot(subplots..., layout = (2, 2), size = (1000, 700))
savefig(p_sweet, "batch_sweet_spot.png")
println("\nSweet spot plot saved to batch_sweet_spot.png")

# --- 4 plots: best batch size per criterion vs static posteriors ---

static_mean_1 = getindex.(static_means, 1)
static_std_1 = getindex.(std.(static_results.posteriors[:x]), 1, 1)
true_state_1 = getindex.(true_states, 1)

comparison_plots = []
for (w, wlabel, c) in zip(speed_weights, weight_labels, colors)
    combined = w .* tr_norm .+ fe_norm
    best_idx = argmin(combined)
    r = results[best_idx]

    p = plot(true_state_1, label = "True state", color = :black, linewidth = 1.5,
        title = "bs=$(r.batch_size) | $(wlabel)",
        xlabel = "t", ylabel = "x₁", legend = :topleft, legendfontsize = 6)
    plot!(static_mean_1, ribbon = 3 .* static_std_1,
        label = "Static (smoothing)", color = :purple, fillalpha = 0.2, linewidth = 1.5)
    plot!(r.posterior_means, ribbon = 3 .* r.posterior_stds,
        label = "Batched (bs=$(r.batch_size))", color = c, fillalpha = 0.3, linewidth = 1.5)

    push!(comparison_plots, p)
end

p_compare = plot(comparison_plots..., layout = (2, 2), size = (1200, 800))
savefig(p_compare, "batch_vs_static.png")
println("Posterior comparison plots saved to batch_vs_static.png")

