#include "constants.h"
#include "custom_distributions.h"
#include "node_detail.h"
#include <algorithm>
#include <common.h>
#include <genetic.h>
#include <iostream>
#include <numeric>
#include <ostream>
#include <philox_engine.h>
#include <program.h>
#include <random>
#include <stack>
#include <array>
#include <span>
#include <string_view>
#include <format>
#include <omp.h>
#include <vector>
#include <memory>
#include <atomic>

namespace genetic {

/**
 * @brief Execute tournaments for all programs with optimized performance.
 *
 * @param progs         Reference to programs vector
 * @param win_indices   Output array for winning indices
 * @param seed          Init seed for random number generation
 * @param n_progs       Number of programs
 * @param n_tours       Number of tournaments to conduct
 * @param tour_size     Number of programs per tournament
 * @param criterion     Selection criterion (0=min, 1=max)
 * @param parsimony     Parsimony coefficient for bloat control
 */
void tournament_kernel(std::span<const program> progs, 
                       std::span<int> win_indices,
                       const int seed, 
                       const int n_tours,
                       const int tour_size, 
                       const bool is_maximization,
                       const float parsimony) {
    const int n_progs = static_cast<int>(progs.size());
    const float sign = is_maximization ? 1.0f : -1.0f;
    
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        PhiloxEngine rng(seed + tid);
        uniform_int_distribution_custom<int> dist(0, n_progs - 1);
        
        #pragma omp for
        for (int idx = 0; idx < n_tours; ++idx) {
            int best_idx = dist(rng);
            float best_score = progs[best_idx].raw_fitness_ - sign * parsimony * progs[best_idx].len;
            
            for (int s = 1; s < tour_size; ++s) {
                int candidate_idx = dist(rng);
                float candidate_score = progs[candidate_idx].raw_fitness_ - sign * parsimony * progs[candidate_idx].len;
                
                // Unified comparison logic using the sign variable
                if (sign * candidate_score > sign * best_score) {
                    best_idx = candidate_idx;
                    best_score = candidate_score;
                }
            }
            
            win_indices[idx] = best_idx;
        }
    }
}

/**
 * @brief Driver function for evolving a generation of programs
 *
 * @param h_oldprogs      Previous generation programs
 * @param h_nextprogs     Next generation programs
 * @param n_samples       Number of samples in dataset
 * @param data            Pointer to input dataset
 * @param y               Pointer to target values
 * @param sample_weights  Pointer to sample weights
 * @param params          Training hyperparameters
 * @param generation      Current generation id
 * @param seed            Random seed for generators
 */
void cpp_evolve(const std::vector<program> &h_oldprogs,
                std::vector<program> &h_nextprogs, 
                int n_samples,
                const float* data, 
                const float* y, 
                const float* sample_weights,
                const param& params, 
                int generation, 
                int seed) {
    const int n_progs = params.population_size;
    const int tour_size = params.tournament_size;
    
    // For first generation, build random programs
    if (generation == 1) {
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            PhiloxEngine local_gen(seed + tid);
            
            #pragma omp for
            for (int i = 0; i < n_progs; ++i) {
                build_program(h_nextprogs[i], params, local_gen);
            }
        }
        
        // Update fitness for the newly generated programs
        set_batched_fitness(n_progs, h_nextprogs, params, n_samples, data, y, sample_weights);
        return;
    }
    
    // Set up mutation probabilities using std::array for better cache locality
    // Pre-compute cumulative probabilities for faster mutation type selection
    std::array<float, 5> cum_probs = {
        params.p_crossover,
        params.p_crossover + params.p_subtree_mutation,
        params.p_crossover + params.p_subtree_mutation + params.p_hoist_mutation,
        params.p_crossover + params.p_subtree_mutation + params.p_hoist_mutation + params.p_point_mutation,
        1.0f
    };
    
    // Count required crossover operations to determine tournaments needed
    int crossover_count = 0;
    
    // Assign mutation types to each program
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        PhiloxEngine local_gen(seed + tid);
        uniform_real_distribution_custom<float> dist(0.0f, 1.0f);
        
        #pragma omp for reduction(+:crossover_count)
        for (int i = 0; i < n_progs; ++i) {
            const float prob = dist(local_gen);
            
            // Use binary search for mutation type selection
            if (prob < cum_probs[0]) {
                h_nextprogs[i].mut_type = mutation_t::crossover;
                crossover_count++;
            } else if (prob < cum_probs[1]) {
                h_nextprogs[i].mut_type = mutation_t::subtree;
            } else if (prob < cum_probs[2]) {
                h_nextprogs[i].mut_type = mutation_t::hoist;
            } else if (prob < cum_probs[3]) {
                h_nextprogs[i].mut_type = mutation_t::point;
            } else {
                h_nextprogs[i].mut_type = mutation_t::reproduce;
            }
        }
    }
    
    // Determine number of tournaments needed
    const int n_tours = n_progs + crossover_count;
    
    // Run tournaments to select parent programs
    std::vector<int> win_indices(n_tours);
    const bool is_maximization = params.criterion() == 1;
    
    tournament_kernel(h_oldprogs, win_indices, seed, n_tours, 
                      tour_size, is_maximization, params.parsimony_coefficient);
    
    // Perform mutations in parallel
    std::atomic<int> donor_pos{n_progs}; // Atomic counter for crossover donor indices
    
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        
        #pragma omp for schedule(dynamic)
        for (int pos = 0; pos < n_progs; ++pos) {
            const int parent_index = win_indices[pos];
            // Create a thread-local engine with good seed diversity
            PhiloxEngine local_gen(seed + tid + pos * 1024);
            
            switch (h_nextprogs[pos].mut_type) {
                case mutation_t::crossover: {
                    // Atomically fetch the next donor index
                    const int donor_index = win_indices[donor_pos.fetch_add(1, std::memory_order_relaxed)];
                    crossover(h_oldprogs[parent_index], h_oldprogs[donor_index],
                              h_nextprogs[pos], params, local_gen);
                    break;
                }
                case mutation_t::subtree:
                    subtree_mutation(h_oldprogs[parent_index], h_nextprogs[pos], params, local_gen);
                    break;
                case mutation_t::hoist:
                    hoist_mutation(h_oldprogs[parent_index], h_nextprogs[pos], params, local_gen);
                    break;
                case mutation_t::point:
                    point_mutation(h_oldprogs[parent_index], h_nextprogs[pos], params, local_gen);
                    break;
                case mutation_t::reproduce:
                    h_nextprogs[pos] = h_oldprogs[parent_index];
                    break;
                default:
                    // Should never happen
                    break;
            }
        }
    }
    
    // Update fitness for the evolved programs
    set_batched_fitness(n_progs, h_nextprogs, params, n_samples, data, y, sample_weights);
}

/**
 * @brief Calculate reproduction probability based on other mutation probabilities
 * @return Float value for reproduce probability
 */
float param::p_reproduce() const {
    return std::clamp(1.0f - (p_crossover + p_subtree_mutation + p_hoist_mutation + p_point_mutation), 
                      0.0f, 1.0f);
}

/**
 * @brief Calculate maximum possible number of programs
 * @return Maximum number of programs
 */
int param::max_programs() const {
    return population_size + generations; // Worst case: one new program per generation
}

/**
 * @brief Determine if metric should be maximized or minimized
 * @return 1 for maximize, 0 for minimize
 */
int param::criterion() const {
    // Return 1 if metric should be maximized, 0 if it should be minimized
    switch (metric) {
        case metric_t::pearson:
        case metric_t::spearman:
            return 1; // Maximize these metrics
        case metric_t::mse:
        case metric_t::logloss:
        case metric_t::mae:
        case metric_t::rmse:
            return 0; // Minimize these metrics
        default:
            return -1; // Error case
    }
}

/**
 * @brief Convert a program to a readable string representation
 * @param prog The program to stringify
 * @return String representation of the program
 */
std::string stringify(const program& prog) {
    if (prog.len == 0) {
        return "()";
    }
    
    std::string result;
    result.reserve(prog.len * 12); // Pre-allocate memory to reduce reallocations
    result += "( ";
    
    std::stack<int> arity_stack;
    arity_stack.push(0);
    
    for (int i = 0; i < prog.len; ++i) {
        const auto& node = prog.nodes[i];
        
        if (node.is_terminal()) {
            // Handle terminal nodes (variables or constants)
            if (i > 0) result += ", ";
            
            if (node.t == node::type::variable) {
                result += "X" + std::to_string(node.u.fid);
            } else {
                result += std::to_string(node.u.val);
            }
            
            // Process closing parentheses
            int remaining = arity_stack.top() - 1;
            arity_stack.pop();
            arity_stack.push(remaining);
            
            while (arity_stack.top() == 0 && arity_stack.size() > 1) {
                arity_stack.pop();
                result += ") ";
                
                if (arity_stack.empty()) break;
                
                remaining = arity_stack.top() - 1;
                arity_stack.pop();
                arity_stack.push(remaining);
            }
        } else {
            // Handle function nodes
            arity_stack.push(node.arity());
            
            if (i > 0) result += ", ";
            
            // Map node type to function name
            switch (node.t) {
                // Binary operators
                case node::type::add:    result += "add("; break;
                case node::type::atan2:  result += "atan2("; break;
                case node::type::div:    result += "div("; break;
                case node::type::fdim:   result += "fdim("; break;
                case node::type::max:    result += "max("; break;
                case node::type::min:    result += "min("; break;
                case node::type::mul:    result += "mult("; break;
                case node::type::pow:    result += "pow("; break;
                case node::type::sub:    result += "sub("; break;
                // Unary operators
                case node::type::abs:    result += "abs("; break;
                case node::type::acos:   result += "acos("; break;
                case node::type::acosh:  result += "acosh("; break;
                case node::type::asin:   result += "asin("; break;
                case node::type::asinh:  result += "asinh("; break;
                case node::type::atan:   result += "atan("; break;
                case node::type::atanh:  result += "atanh("; break;
                case node::type::cbrt:   result += "cbrt("; break;
                case node::type::cos:    result += "cos("; break;
                case node::type::cosh:   result += "cosh("; break;
                case node::type::cube:   result += "cube("; break;
                case node::type::exp:    result += "exp("; break;
                case node::type::inv:    result += "inv("; break;
                case node::type::log:    result += "log("; break;
                case node::type::neg:    result += "neg("; break;
                case node::type::rcbrt:  result += "rcbrt("; break;
                case node::type::rsqrt:  result += "rsqrt("; break;
                case node::type::sin:    result += "sin("; break;
                case node::type::sinh:   result += "sinh("; break;
                case node::type::sq:     result += "sq("; break;
                case node::type::sqrt:   result += "sqrt("; break;
                case node::type::tan:    result += "tan("; break;
                case node::type::tanh:   result += "tanh("; break;
                default: break;
            }
            result += " ";
        }
    }
    
    // Close any remaining parentheses
    while (arity_stack.size() > 1) {
        arity_stack.pop();
        result += ") ";
    }
    
    result += ")";
    return result;
}

/**
 * @brief Symbolic regression fitting function
 */
void symFit(const float* input, 
            const float* labels,
            const float* sample_weights, 
            const int n_rows, 
            const int n_cols,
            param& params, 
            program_t& final_progs,
            std::vector<std::vector<program>>& history) {
    // Update arity map in params
    for (auto function_type : params.function_set) {
        // Determine arity based on function type
        int arity = (node::type::binary_begin <= function_type && 
                     function_type <= node::type::binary_end) ? 2 : 1;
        
        // Use C++17 map::try_emplace for efficiency
        auto [it, inserted] = params.arity_set.try_emplace(arity, std::vector<node::type>{function_type});
        
        if (!inserted) {
            auto& functions = it->second;
            if (std::find(functions.begin(), functions.end(), function_type) == functions.end()) {
                functions.push_back(function_type);
            }
        }
    }
    
    // Set terminal ratio dynamically if needed
    const bool dynamic_terminal_ratio = (params.terminalRatio == 0.0f);
    if (dynamic_terminal_ratio) {
        params.terminalRatio = static_cast<float>(params.num_features) / 
                              (params.num_features + params.function_set.size());
    }
    
    /* Initializations */
    std::vector<program> h_currprogs(params.population_size);
    std::vector<program> h_nextprogs(params.population_size);
    PhiloxEngine h_gen_engine(params.random_state);
    uniform_int_distribution_custom<int> seed_dist;
    
    /* Begin training loop */
    params.num_epochs = 0;
    const bool is_minimization = params.criterion() == 0;
    const float target_fitness = params.stopping_criteria;
    
    for (int gen = 0; gen < params.generations; ++gen) {
        // Generate seed for this generation
        const int init_seed = seed_dist(h_gen_engine);
        
        // Evolve population
        cpp_evolve(h_currprogs, h_nextprogs, n_rows, input, labels, sample_weights,
                   params, gen + 1, init_seed);
        
        ++params.num_epochs;
        
        // Swap populations
        h_currprogs.swap(h_nextprogs);
        
        // Maintain evolution history with minimal memory usage
        if (!params.low_memory || gen == 0) {
            history.push_back(h_currprogs);
        } else {
            history.back() = h_currprogs;
        }
        
        // Find optimal fitness using parallel reduction
        float opt_fit = is_minimization ? 
            std::numeric_limits<float>::max() : 
            std::numeric_limits<float>::lowest();
        
        #pragma omp parallel
        {
            float thread_best = opt_fit;
            
            #pragma omp for nowait
            for (int i = 0; i < params.population_size; ++i) {
                if (is_minimization) {
                    thread_best = std::min(thread_best, h_currprogs[i].raw_fitness_);
                } else {
                    thread_best = std::max(thread_best, h_currprogs[i].raw_fitness_);
                }
            }
            
            #pragma omp critical
            {
                if (is_minimization) {
                    opt_fit = std::min(opt_fit, thread_best);
                } else {
                    opt_fit = std::max(opt_fit, thread_best);
                }
            }
        }
        
        // Check early stopping criteria
        if ((is_minimization && opt_fit <= target_fitness) ||
            (!is_minimization && opt_fit >= target_fitness)) {
            std::cerr << "Early stopping at Generation #" << (gen + 1)
                      << ", fitness=" << opt_fit << std::endl;
            break;
        }
    }
    
    // Copy final programs
    std::copy(h_currprogs.begin(), h_currprogs.end(), final_progs);
    
    // Reset automatic growth parameter if it was set dynamically
    if (dynamic_terminal_ratio) {
        params.terminalRatio = 0.0f;
    }
}

/**
 * @brief Predict using symbolic regression model
 */
void symRegPredict(const float* input, 
                   const int n_rows,
                   const program_t& best_prog, 
                   float* output) {
    execute(best_prog, n_rows, 1, input, output);
}

/**
 * @brief Predict probabilities for symbolic classification
 */
void symClfPredictProbs(const float* input, 
                        int n_rows,
                        const param& params, 
                        const program_t& best_prog,
                        float* output) {
    // Execute program to get raw predictions
    execute(best_prog, n_rows, 1, input, output);
    
    // Apply sigmoid transformation if needed
    if (params.transformer == transformer_t::sigmoid) {
        #pragma omp parallel for
        for (int i = 0; i < n_rows; ++i) {
            const float raw_prediction = output[i];
            const float sigmoid_value = 1.0f / (1.0f + std::exp(-raw_prediction));
            
            // Store probabilities for both classes
            output[i + n_rows] = sigmoid_value;        // Probability of class 1
            output[i] = 1.0f - sigmoid_value;          // Probability of class 0
        }
    }
}

/**
 * @brief Predict classes for symbolic classification
 */
void symClfPredict(const float* input, 
                   int n_rows, 
                   const param& params,
                   const program_t& best_prog, 
                   float* output) {
    // Allocate memory for probabilities
    std::vector<float> probs(2 * n_rows);
    
    // Get class probabilities
    symClfPredictProbs(input, n_rows, params, best_prog, probs.data());
    
    // Determine class prediction based on probability comparison
    #pragma omp parallel for
    for (int i = 0; i < n_rows; ++i) {
        output[i] = (probs[i] <= probs[i + n_rows]) ? 1.0f : 0.0f;
    }
}

/**
 * @brief Transform data using symbolic programs
 */
void symTransform(const float* input, 
                  const param& params,
                  const program_t& final_progs, 
                  const int n_rows,
                  const int n_cols, 
                  float* output) {
    // Execute multiple programs on input data
    execute(final_progs, n_rows, params.n_components, input, output);
}

} // namespace genetic
// #include "constants.h"
// #include "custom_distributions.h"
// #include "node_detail.h"
// #include <algorithm>
// #include <common.h>
// #include <genetic.h>
// #include <iostream>
// #include <numeric>
// #include <ostream>
// #include <philox_engine.h>
// #include <program.h>
// #include <random>
// #include <stack>
// #include <array>
// #include <omp.h>
// namespace genetic {
//
// /**
//  * @brief Execute tournaments for all programs.
//  *        The fitness values being compared are adjusted for bloat (program
//  *        length), using the given parsimony coefficient.
//  *
//  * @param progs         Device pointer to programs
//  * @param win_indices   Winning indices for every tournament
//  * @param seeds         Init seeds for choice selection
//  * @param n_progs       Number of programs
//  * @param n_tours       No of tournaments to be conducted
//  * @param tour_size     No of programs considered per tournament(@c <=n_progs><)
//  * @param criterion     Selection criterion for choices(min/max)
//  * @param parsimony     Parsimony coefficient to account for bloat
//  */
// void tournament_kernel(const std::vector<program> &progs, int *win_indices,
//                        const int seed, const int n_progs, const int n_tours,
//                        const int tour_size, const int criterion,
//                        const float parsimony) {
//   const float sign = static_cast<float>(2 * criterion - 1);
//   #pragma omp parallel
//   {
//     int tid = omp_get_thread_num();
//     PhiloxEngine local_gen(seed + tid);
//     uniform_int_distribution_custom<int> rng(0, n_progs - 1);
//     #pragma omp for
//     for (int idx = 0; idx < n_tours; ++idx) {
//       int opt = rng(local_gen);
//       float opt_score = progs[opt].raw_fitness_ - parsimony * progs[opt].len * sign;
//       if (criterion) {
//         // Maximization: choose candidate with highest adjusted fitness.
//         for (int s = 1; s < tour_size; ++s) {
//           int curr = rng(local_gen);
//           float curr_score = progs[curr].raw_fitness_ - parsimony * progs[curr].len * sign;
//           if (curr_score > opt_score) {
//             opt = curr;
//             opt_score = curr_score;
//           }
//         }
//       } else {
//         // Minimization: choose candidate with lowest adjusted fitness.
//         for (int s = 1; s < tour_size; ++s) {
//           int curr = rng(local_gen);
//           float curr_score = progs[curr].raw_fitness_ - parsimony * progs[curr].len * sign;
//           if (curr_score < opt_score) {
//             opt = curr;
//             opt_score = curr_score;
//           }
//         }
//       }
//       win_indices[idx] = opt;
//     }
//   }
// }
//
// /**
//  * @brief Driver function for evolving a generation of programs
//  *
//  * @param h_oldprogs      previous generation host programs
//  * @param h_nextprogs     next generation host programs
//  * @param n_samples       No of samples in input dataset
//  * @param data            Device pointer to input dataset
//  * @param y               Device pointer to input predictions
//  * @param sample_weights  Device pointer to input weights
//  * @param params          Training hyperparameters
//  * @param generation      Current generation id
//  * @param seed            Random seed for generators
//  */
// void cpp_evolve(const std::vector<program> &h_oldprogs,
//                 std::vector<program> &h_nextprogs, int n_samples,
//                 const float *data, const float *y, const float *sample_weights,
//                 const param &params, int generation, int seed) {
//   int n_progs = params.population_size;
//   int tour_size = params.tournament_size;
//   int n_tours = n_progs; // at least one tournament per program
//
//   // For generation 1, build random programs
//   if (generation == 1) {
//     #pragma omp parallel
//     {
//       int tid = omp_get_thread_num();
//       PhiloxEngine local_gen(seed + tid);
//       #pragma omp for
//       for (int i = 0; i < n_progs; ++i) {
//         build_program(h_nextprogs[i], params, local_gen);
//       }
//     }
//   } else {
//     // Set up mutation probabilities; use std::array and partial_sum for cumulative probabilities.
//     std::array<float, 4> mut_probs = { params.p_crossover, params.p_subtree_mutation,
//                                         params.p_hoist_mutation, params.p_point_mutation };
//     std::partial_sum(mut_probs.begin(), mut_probs.end(), mut_probs.begin());
//     int crossover_count = 0;
//     // Decide mutation type per program using a thread-local RNG.
//     #pragma omp parallel
//     {
//       int tid = omp_get_thread_num();
//       PhiloxEngine local_gen(seed + tid);
//       uniform_real_distribution_custom<float> dist(0.0f, 1.0f);
//       #pragma omp for reduction(+:crossover_count)
//       for (int i = 0; i < n_progs; ++i) {
//         float prob = dist(local_gen);
//         if (prob < mut_probs[0]) {
//           h_nextprogs[i].mut_type = mutation_t::crossover;
//           crossover_count++;
//         } else if (prob < mut_probs[1]) {
//           h_nextprogs[i].mut_type = mutation_t::subtree;
//         } else if (prob < mut_probs[2]) {
//           h_nextprogs[i].mut_type = mutation_t::hoist;
//         } else if (prob < mut_probs[3]) {
//           h_nextprogs[i].mut_type = mutation_t::point;
//         } else {
//           h_nextprogs[i].mut_type = mutation_t::reproduce;
//         }
//       }
//     }
//     n_tours += crossover_count;
//
//     // Run tournaments to choose candidate indices.
//     std::vector<int> win_indices(n_tours);
//     int crit = params.criterion();
//     tournament_kernel(h_oldprogs, win_indices.data(), seed, n_progs, n_tours, tour_size, crit,
//                       params.parsimony_coefficient);
//
//     // Perform host mutations.
//     int donor_pos = n_progs; // donor indices start after first n_progs
//     #pragma omp parallel for schedule(dynamic)
//     for (int pos = 0; pos < n_progs; ++pos) {
//       int parent_index = win_indices[pos];
//       int tid = omp_get_thread_num();
//       // Create a thread-local engine per iteration (offset by pos for diversity)
//       PhiloxEngine local_gen(seed + tid + pos);
//       switch (h_nextprogs[pos].mut_type) {
//         case mutation_t::crossover: {
//           int donor_index;
//           #pragma omp critical
//           {
//             donor_index = win_indices[donor_pos++];
//           }
//           crossover(h_oldprogs[parent_index], h_oldprogs[donor_index],
//                     h_nextprogs[pos], params, local_gen);
//           break;
//         }
//         case mutation_t::subtree:
//           subtree_mutation(h_oldprogs[parent_index], h_nextprogs[pos], params, local_gen);
//           break;
//         case mutation_t::hoist:
//           hoist_mutation(h_oldprogs[parent_index], h_nextprogs[pos], params, local_gen);
//           break;
//         case mutation_t::point:
//           point_mutation(h_oldprogs[parent_index], h_nextprogs[pos], params, local_gen);
//           break;
//         case mutation_t::reproduce:
//           h_nextprogs[pos] = h_oldprogs[parent_index];
//           break;
//         default:
//           break;
//       }
//     }
//   }
//
//   // Update fitness for the evolved programs.
//   set_batched_fitness(n_progs, h_nextprogs, params, n_samples, data, y, sample_weights);
// }
// float param::p_reproduce() const {
//   float sum = p_crossover + p_subtree_mutation + p_hoist_mutation + p_point_mutation;
//   return std::clamp(1.f - sum, 0.f, 1.f);
// }
//
// int param::max_programs() const {
//   // in the worst case every generation's top program ends up reproducing,
//   // thereby adding another program into the population
//   return this->population_size + this->generations;
// }
//
// int param::criterion() const {
//   // Returns 0 if a smaller value is preferred and 1 for the opposite
//   switch (this->metric) {
//   case metric_t::mse:
//     return 0;
//   case metric_t::logloss:
//     return 0;
//   case metric_t::mae:
//     return 0;
//   case metric_t::rmse:
//     return 0;
//   case metric_t::pearson:
//     return 1;
//   case metric_t::spearman:
//     return 1;
//   default:
//     return -1;
//   }
// }
//
// std::string stringify(const program &prog) {
//   std::string eqn = "( ";
//   std::string delim = "";
//   std::stack<int> ar_stack;
//   ar_stack.push(0);
//
//   for (int i = 0; i < prog.len; ++i) {
//     if (prog.nodes[i].is_terminal()) {
//       eqn += delim;
//       if (prog.nodes[i].t == node::type::variable) {
//         // variable
//         eqn += "X";
//         eqn += std::to_string(prog.nodes[i].u.fid);
//       } else {
//         // const
//         eqn += std::to_string(prog.nodes[i].u.val);
//       }
//
//       int end_elem = ar_stack.top();
//       ar_stack.pop();
//       ar_stack.push(end_elem - 1);
//       while (ar_stack.top() == 0) {
//         ar_stack.pop();
//         eqn += ") ";
//         if (ar_stack.empty()) {
//           break;
//         }
//         end_elem = ar_stack.top();
//         ar_stack.pop();
//         ar_stack.push(end_elem - 1);
//       }
//       delim = ", ";
//     } else {
//       ar_stack.push(prog.nodes[i].arity());
//       eqn += delim;
//       switch (prog.nodes[i].t) {
//       // binary operators
//       case node::type::add:
//         eqn += "add(";
//         break;
//       case node::type::atan2:
//         eqn += "atan2(";
//         break;
//       case node::type::div:
//         eqn += "div(";
//         break;
//       case node::type::fdim:
//         eqn += "fdim(";
//         break;
//       case node::type::max:
//         eqn += "max(";
//         break;
//       case node::type::min:
//         eqn += "min(";
//         break;
//       case node::type::mul:
//         eqn += "mult(";
//         break;
//       case node::type::pow:
//         eqn += "pow(";
//         break;
//       case node::type::sub:
//         eqn += "sub(";
//         break;
//       // unary operators
//       case node::type::abs:
//         eqn += "abs(";
//         break;
//       case node::type::acos:
//         eqn += "acos(";
//         break;
//       case node::type::acosh:
//         eqn += "acosh(";
//         break;
//       case node::type::asin:
//         eqn += "asin(";
//         break;
//       case node::type::asinh:
//         eqn += "asinh(";
//         break;
//       case node::type::atan:
//         eqn += "atan(";
//         break;
//       case node::type::atanh:
//         eqn += "atanh(";
//         break;
//       case node::type::cbrt:
//         eqn += "cbrt(";
//         break;
//       case node::type::cos:
//         eqn += "cos(";
//         break;
//       case node::type::cosh:
//         eqn += "cosh(";
//         break;
//       case node::type::cube:
//         eqn += "cube(";
//         break;
//       case node::type::exp:
//         eqn += "exp(";
//         break;
//       case node::type::inv:
//         eqn += "inv(";
//         break;
//       case node::type::log:
//         eqn += "log(";
//         break;
//       case node::type::neg:
//         eqn += "neg(";
//         break;
//       case node::type::rcbrt:
//         eqn += "rcbrt(";
//         break;
//       case node::type::rsqrt:
//         eqn += "rsqrt(";
//         break;
//       case node::type::sin:
//         eqn += "sin(";
//         break;
//       case node::type::sinh:
//         eqn += "sinh(";
//         break;
//       case node::type::sq:
//         eqn += "sq(";
//         break;
//       case node::type::sqrt:
//         eqn += "sqrt(";
//         break;
//       case node::type::tan:
//         eqn += "tan(";
//         break;
//       case node::type::tanh:
//         eqn += "tanh(";
//         break;
//       default:
//         break;
//       }
//       eqn += " ";
//       delim = "";
//     }
//   }
//
//   eqn += ")";
//   return eqn;
// }
//
// void symFit(const float *input, const float *labels,
//             const float *sample_weights, const int n_rows, const int n_cols,
//             param &params, program_t &final_progs,
//             std::vector<std::vector<program>> &history) {
//
//   // Update arity map in params - Need to do this only here, as all operations
//   // will call Fit at least once
//   for (auto f : params.function_set) {
//     int ar = 1;
//     if (node::type::binary_begin <= f && f <= node::type::binary_end) {
//       ar = 2;
//     }
//
//     if (params.arity_set.find(ar) == params.arity_set.end()) {
//       // Create map entry for current arity
//       std::vector<node::type> vec_f(1, f);
//       params.arity_set.insert(std::make_pair(ar, vec_f));
//     } else {
//       // Insert into map
//       std::vector<node::type> vec_f = params.arity_set.at(ar);
//       if (std::find(vec_f.begin(), vec_f.end(), f) == vec_f.end()) {
//         params.arity_set.at(ar).push_back(f);
//       }
//     }
//   }
//
//   // Check terminalRatio to dynamically set it
//   bool growAuto = (params.terminalRatio == 0.0f);
//   if (growAuto) {
//     params.terminalRatio = 1.0f * params.num_features /
//                            (params.num_features + params.function_set.size());
//   }
//
//   /* Initializations */
//
//   std::vector<program> h_currprogs(params.population_size);
//   std::vector<program> h_nextprogs(params.population_size);
//
//   std::vector<float> h_fitness(params.population_size, 0.0f);
//
//   PhiloxEngine h_gen_engine(params.random_state);
//   ;
//   uniform_int_distribution_custom<int> seed_dist;
//
//   /* Begin training */
//   auto gen = 0;
//   params.num_epochs = 0;
//
//   while (gen < params.generations) {
//     int init_seed = seed_dist(h_gen_engine);
//     cpp_evolve(h_currprogs, h_nextprogs, n_rows, input, labels, sample_weights,
//                params, gen + 1, init_seed);
//     ++params.num_epochs;
//     h_currprogs = h_nextprogs;
//
//     // Maintain evolution history.
//     if (!params.low_memory || gen == 0)
//       history.push_back(h_currprogs);
//     else
//       history.back() = h_currprogs;
//
//     // Compute optimal fitness using a parallel reduction.
//     int crit = params.criterion();
//     float opt_fit;
//     if (crit == 0) {
//       opt_fit = std::numeric_limits<float>::max();
//       #pragma omp parallel for reduction(min:opt_fit)
//       for (int i = 0; i < params.population_size; ++i)
//         opt_fit = std::min(opt_fit, h_currprogs[i].raw_fitness_);
//     } else {
//       opt_fit = std::numeric_limits<float>::lowest();
//       #pragma omp parallel for reduction(max:opt_fit)
//       for (int i = 0; i < params.population_size; ++i)
//         opt_fit = std::max(opt_fit, h_currprogs[i].raw_fitness_);
//     }
//     if ((crit == 0 && opt_fit <= params.stopping_criteria) ||
//         (crit == 1 && opt_fit >= params.stopping_criteria)) {
//       std::cerr << "Early stopping at Generation #" << (gen + 1)
//                 << ", fitness=" << opt_fit << std::endl;
//       break;
//     }
//     ++gen;
//   }  // Set final generation programs(deepcopy)
//   std::copy(h_currprogs.begin(), h_currprogs.end(), final_progs);
//
//   // Reset automatic growth parameter
//   if (growAuto) {
//     params.terminalRatio = 0.0f;
//   }
// }
//
// void symRegPredict(const float *input, const int n_rows,
//                    const program_t &best_prog, float *output) {
//   // Assume best_prog is on device
//   execute(best_prog, n_rows, 1, input, output);
// }
//
// void symClfPredictProbs(const float* input, int n_rows,
//                         const param& params, const program_t& best_prog,
//                         float* output) {
//   execute(best_prog, n_rows, 1, input, output);
//   if (params.transformer == transformer_t::sigmoid) {
//     #pragma omp parallel for
//     for (int i = 0; i < n_rows; ++i) {
//       float in = output[i];
//       float sigmoid = 1.0f / (1.0f + expf(-in));
//       output[i + n_rows] = sigmoid;
//       output[i] = 1.0f - sigmoid;
//     }
//   }
// }
//
//
// void symClfPredict(const float* input, int n_rows, const param& params,
//                    const program_t& best_prog, float* output) {
//   std::vector<float> probs(2 * n_rows);
//   symClfPredictProbs(input, n_rows, params, best_prog, probs.data());
//   #pragma omp parallel for
//   for (int i = 0; i < n_rows; ++i)
//     output[i] = (probs[i] <= probs[i + n_rows]) ? 1.0f : 0.0f;
// }
//
// void symTransform(const float *input, const param &params,
//                   const program_t &final_progs, const int n_rows,
//                   const int n_cols, float *output) {
//   // cudaStream_t stream = handle.get_stream();
//   // Execute final_progs(ordered by fitness) on input
//   // output of size [n_rows,hall_of_fame]
//   execute(final_progs, n_rows, params.n_components, input, output);
// }
//
// } // namespace genetic
