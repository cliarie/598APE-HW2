
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
#include <omp.h>
namespace genetic {

/**
 * @brief Execute tournaments for all programs.
 *        The fitness values being compared are adjusted for bloat (program
 *        length), using the given parsimony coefficient.
 *
 * @param progs         Device pointer to programs
 * @param win_indices   Winning indices for every tournament
 * @param seeds         Init seeds for choice selection
 * @param n_progs       Number of programs
 * @param n_tours       No of tournaments to be conducted
 * @param tour_size     No of programs considered per tournament(@c <=n_progs><)
 * @param criterion     Selection criterion for choices(min/max)
 * @param parsimony     Parsimony coefficient to account for bloat
 */
void tournament_kernel(const std::vector<program> &progs, int *win_indices,
                       const int seed, const int n_progs, const int n_tours,
                       const int tour_size, const int criterion,
                       const float parsimony) {
  #pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    PhiloxEngine thread_gen(seed + thread_id);

    #pragma omp for
    for (auto idx = 0; idx < n_tours; ++idx) {
      uniform_int_distribution_custom<int> thread_rng(0, n_progs-1);
      int r = thread_rng(thread_gen);
      // Define optima values
      int opt = r % n_progs;
      float opt_penalty = parsimony * progs[opt].len * (2 * criterion - 1);
      float opt_score = progs[opt].raw_fitness_ - opt_penalty;

      for (int s = 1; s < tour_size; ++s) {
        r = thread_rng(thread_gen);
        int curr = r % n_progs;
        float curr_penalty = parsimony * progs[curr].len * (2 * criterion - 1);
        float curr_score = progs[curr].raw_fitness_ - curr_penalty;

        if ((criterion && (opt_score < curr_score)) || (!criterion && (opt_score > curr_score))) {
          opt = curr;
          opt_penalty = curr_penalty;
          opt_score = curr_score;
        }
      }
      // Set win index
      win_indices[idx] = opt;
    }
  }
}

/**
 * @brief Driver function for evolving a generation of programs
 *
 * @param h_oldprogs      previous generation host programs
 * @param h_nextprogs     next generation host programs
 * @param n_samples       No of samples in input dataset
 * @param data            Device pointer to input dataset
 * @param y               Device pointer to input predictions
 * @param sample_weights  Device pointer to input weights
 * @param params          Training hyperparameters
 * @param generation      Current generation id
 * @param seed            Random seed for generators
 */
void cpp_evolve(const std::vector<program> &h_oldprogs,
                std::vector<program> &h_nextprogs, const int n_samples,
                const float *data, const float *y, const float *sample_weights,
                const param &params, const int generation, const int seed) {
  auto n_progs = params.population_size;
  auto tour_size = params.tournament_size;
  auto n_tours = n_progs; // at least num_progs tournaments

  // Seed engines
  PhiloxEngine h_gen(seed); // use Philox instead of Mersenne Twister?

  uniform_real_distribution_custom<float> dist_U(0.0f, 1.0f);

  // Build, Mutate and Run Tournaments

  if (generation == 1) {
    // Build random programs for the first generation
    #pragma omp parallel for
    for (auto i = 0; i < n_progs; ++i) {
      build_program(h_nextprogs[i], params, h_gen);
    }

  } else {
    // Set mutation type
    float mut_probs[4];
    mut_probs[0] = params.p_crossover;
    mut_probs[1] = params.p_subtree_mutation;
    mut_probs[2] = params.p_hoist_mutation;
    mut_probs[3] = params.p_point_mutation;
    std::partial_sum(mut_probs, mut_probs + 4, mut_probs);

    int crossover_count = 0;
    #pragma omp parallel for reduction(+:crossover_count)
    for (auto i = 0; i < n_progs; ++i) {
      float prob = dist_U(h_gen);

      if (prob < mut_probs[0]) {
        h_nextprogs[i].mut_type = mutation_t::crossover;
        crossover_count++; // each thread adds own count
      } else if (prob < mut_probs[1]) {
        h_nextprogs[i].mut_type = mutation_t::subtree;
      } else if (prob < mut_probs[2]) {
        h_nextprogs[i].mut_type = mutation_t::hoist;
      } else if (prob < mut_probs[3]) {
        h_nextprogs[i].mut_type = mutation_t::point;
      } else {
        h_nextprogs[i].mut_type = mutation_t::reproduce;
      }
    }
    n_tours += crossover_count; // update safely outside parallel

    // Run tournaments
    std::vector<int> d_win_indices(n_tours);

    auto criterion = params.criterion();
    tournament_kernel(h_oldprogs, d_win_indices.data(), seed, n_progs, n_tours,
                      tour_size, criterion, params.parsimony_coefficient);

    // dim3 nblks(raft::ceildiv(n_tours, GENE_TPB), 1, 1);
    // batched_tournament_kernel<<<nblks, GENE_TPB, 0, stream>>>(
    //     d_oldprogs, d_win_indices.data(), tour_seeds.data(), n_progs,
    //     n_tours, tour_size, criterion, params.parsimony_coefficient);
    // RAFT_CUDA_TRY(cudaPeekAtLastError());
    // h.sync_stream(stream);

    // Perform host mutations

    auto donor_pos = n_progs;

    #pragma omp parallel for schedule(dynamic)
    for (auto pos = 0; pos < n_progs; ++pos) {
      auto parent_index = d_win_indices[pos];

      if (h_nextprogs[pos].mut_type == mutation_t::crossover) {
        // Get secondary index
        int donor_index;

        #pragma omp critical
        {
          donor_index = d_win_indices[donor_pos];
          donor_pos++;
        }
        crossover(h_oldprogs[parent_index], h_oldprogs[donor_index],
                  h_nextprogs[pos], params, h_gen);
      } else if (h_nextprogs[pos].mut_type == mutation_t::subtree) {
        subtree_mutation(h_oldprogs[parent_index], h_nextprogs[pos], params,
                         h_gen);
      } else if (h_nextprogs[pos].mut_type == mutation_t::hoist) {
        hoist_mutation(h_oldprogs[parent_index], h_nextprogs[pos], params,
                       h_gen);
      } else if (h_nextprogs[pos].mut_type == mutation_t::point) {
        point_mutation(h_oldprogs[parent_index], h_nextprogs[pos], params,
                       h_gen);
      } else if (h_nextprogs[pos].mut_type == mutation_t::reproduce) {
        h_nextprogs[pos] = h_oldprogs[parent_index];
      } else {
        // Should not come here
      }
    }
  }

  // Update raw fitness for all programs
  set_batched_fitness(n_progs, h_nextprogs, params, n_samples, data, y,
                      sample_weights);
}

float param::p_reproduce() const {
  auto sum = this->p_crossover + this->p_subtree_mutation +
             this->p_hoist_mutation + this->p_point_mutation;
  auto ret = 1.f - sum;
  return fmaxf(0.f, fminf(ret, 1.f));
}

int param::max_programs() const {
  // in the worst case every generation's top program ends up reproducing,
  // thereby adding another program into the population
  return this->population_size + this->generations;
}

int param::criterion() const {
  // Returns 0 if a smaller value is preferred and 1 for the opposite
  switch (this->metric) {
  case metric_t::mse:
    return 0;
  case metric_t::logloss:
    return 0;
  case metric_t::mae:
    return 0;
  case metric_t::rmse:
    return 0;
  case metric_t::pearson:
    return 1;
  case metric_t::spearman:
    return 1;
  default:
    return -1;
  }
}

std::string stringify(const program &prog) {
  std::string eqn = "( ";
  std::string delim = "";
  std::stack<int> ar_stack;
  ar_stack.push(0);

  for (int i = 0; i < prog.len; ++i) {
    if (prog.nodes[i].is_terminal()) {
      eqn += delim;
      if (prog.nodes[i].t == node::type::variable) {
        // variable
        eqn += "X";
        eqn += std::to_string(prog.nodes[i].u.fid);
      } else {
        // const
        eqn += std::to_string(prog.nodes[i].u.val);
      }

      int end_elem = ar_stack.top();
      ar_stack.pop();
      ar_stack.push(end_elem - 1);
      while (ar_stack.top() == 0) {
        ar_stack.pop();
        eqn += ") ";
        if (ar_stack.empty()) {
          break;
        }
        end_elem = ar_stack.top();
        ar_stack.pop();
        ar_stack.push(end_elem - 1);
      }
      delim = ", ";
    } else {
      ar_stack.push(prog.nodes[i].arity());
      eqn += delim;
      switch (prog.nodes[i].t) {
      // binary operators
      case node::type::add:
        eqn += "add(";
        break;
      case node::type::atan2:
        eqn += "atan2(";
        break;
      case node::type::div:
        eqn += "div(";
        break;
      case node::type::fdim:
        eqn += "fdim(";
        break;
      case node::type::max:
        eqn += "max(";
        break;
      case node::type::min:
        eqn += "min(";
        break;
      case node::type::mul:
        eqn += "mult(";
        break;
      case node::type::pow:
        eqn += "pow(";
        break;
      case node::type::sub:
        eqn += "sub(";
        break;
      // unary operators
      case node::type::abs:
        eqn += "abs(";
        break;
      case node::type::acos:
        eqn += "acos(";
        break;
      case node::type::acosh:
        eqn += "acosh(";
        break;
      case node::type::asin:
        eqn += "asin(";
        break;
      case node::type::asinh:
        eqn += "asinh(";
        break;
      case node::type::atan:
        eqn += "atan(";
        break;
      case node::type::atanh:
        eqn += "atanh(";
        break;
      case node::type::cbrt:
        eqn += "cbrt(";
        break;
      case node::type::cos:
        eqn += "cos(";
        break;
      case node::type::cosh:
        eqn += "cosh(";
        break;
      case node::type::cube:
        eqn += "cube(";
        break;
      case node::type::exp:
        eqn += "exp(";
        break;
      case node::type::inv:
        eqn += "inv(";
        break;
      case node::type::log:
        eqn += "log(";
        break;
      case node::type::neg:
        eqn += "neg(";
        break;
      case node::type::rcbrt:
        eqn += "rcbrt(";
        break;
      case node::type::rsqrt:
        eqn += "rsqrt(";
        break;
      case node::type::sin:
        eqn += "sin(";
        break;
      case node::type::sinh:
        eqn += "sinh(";
        break;
      case node::type::sq:
        eqn += "sq(";
        break;
      case node::type::sqrt:
        eqn += "sqrt(";
        break;
      case node::type::tan:
        eqn += "tan(";
        break;
      case node::type::tanh:
        eqn += "tanh(";
        break;
      default:
        break;
      }
      eqn += " ";
      delim = "";
    }
  }

  eqn += ")";
  return eqn;
}

void symFit(const float *input, const float *labels,
            const float *sample_weights, const int n_rows, const int n_cols,
            param &params, program_t &final_progs,
            std::vector<std::vector<program>> &history) {

  // Update arity map in params - Need to do this only here, as all operations
  // will call Fit at least once
  for (auto f : params.function_set) {
    int ar = 1;
    if (node::type::binary_begin <= f && f <= node::type::binary_end) {
      ar = 2;
    }

    if (params.arity_set.find(ar) == params.arity_set.end()) {
      // Create map entry for current arity
      std::vector<node::type> vec_f(1, f);
      params.arity_set.insert(std::make_pair(ar, vec_f));
    } else {
      // Insert into map
      std::vector<node::type> vec_f = params.arity_set.at(ar);
      if (std::find(vec_f.begin(), vec_f.end(), f) == vec_f.end()) {
        params.arity_set.at(ar).push_back(f);
      }
    }
  }

  // Check terminalRatio to dynamically set it
  bool growAuto = (params.terminalRatio == 0.0f);
  if (growAuto) {
    params.terminalRatio = 1.0f * params.num_features /
                           (params.num_features + params.function_set.size());
  }

  /* Initializations */

  std::vector<program> h_currprogs(params.population_size);
  std::vector<program> h_nextprogs(params.population_size);

  std::vector<float> h_fitness(params.population_size, 0.0f);

  PhiloxEngine h_gen_engine(params.random_state);
  ;
  uniform_int_distribution_custom<int> seed_dist;

  /* Begin training */
  auto gen = 0;
  params.num_epochs = 0;

  while (gen < params.generations) {
    // Generate an init seed
    auto init_seed = seed_dist(h_gen_engine);

    // Evolve current generation
    cpp_evolve(h_currprogs, h_nextprogs, n_rows, input, labels, sample_weights,
               params, (gen + 1), init_seed);
    // Update epochs
    ++params.num_epochs;

    // Update h_currprogs (deepcopy)
    h_currprogs = h_nextprogs;

    // Update evolution history, depending on the low memory flag
    if (!params.low_memory || gen == 0) {
      history.push_back(h_currprogs);
    } else {
      history.back() = h_currprogs;
    }

    // Update fitness array [host] and compute stopping criterion
    auto crit = params.criterion();
    h_fitness[0] = h_currprogs[0].raw_fitness_;
    auto opt_fit = h_fitness[0];

    // #pragma omp parallel for reduction(min:opt_fit) if (crit == 0) reduction(max:opt_fit) if (crit == 1)
    for (auto i = 1; i < params.population_size; ++i) {
      h_fitness[i] = h_currprogs[i].raw_fitness_;

      if (crit == 0) {
        opt_fit = std::min(opt_fit, h_fitness[i]);
      } else {
        opt_fit = std::max(opt_fit, h_fitness[i]);
      }
    }

    // Check for stop criterion
    if ((crit == 0 && opt_fit <= params.stopping_criteria) ||
        (crit == 1 && opt_fit >= params.stopping_criteria)) {
      std::cerr << "Early stopping criterion reached in Generation #"
                << (gen + 1) << ", fitness=" << opt_fit << std::endl;
      break;
    }

    // Update generation
    ++gen;
  }

  // Set final generation programs(deepcopy)
  std::copy(h_currprogs.begin(), h_currprogs.end(), final_progs);

  // Reset automatic growth parameter
  if (growAuto) {
    params.terminalRatio = 0.0f;
  }
}

void symRegPredict(const float *input, const int n_rows,
                   const program_t &best_prog, float *output) {
  // Assume best_prog is on device
  execute(best_prog, n_rows, 1, input, output);
}

void symClfPredictProbs(const float *input, const int n_rows,
                        const param &params, const program_t &best_prog,
                        float *output) {

  // Assume output is of shape [n_rows, 2] in colMajor format
  execute(best_prog, n_rows, 1, input, output);

  // Apply 2 map operations
  if (params.transformer == transformer_t::sigmoid) {
    for (auto i = 0; i < n_rows; ++i) {
      float in = output[i];
      output[i + n_rows] = 1.0f / (1.0f + expf(-in));
      output[i] = 1.0f - output[i + n_rows];
    }
  } else {
    // Only sigmoid supported for now
  }
}

void symClfPredict(const float *input, const int n_rows, const param &params,
                   const program_t &best_prog, float *output) {

  // Memory for probabilities
  std::vector<float> probs(2 * n_rows);
  symClfPredictProbs(input, n_rows, params, best_prog, probs.data());

  // Take argmax along columns
  // TODO: Further modification needed for n_classes
  for (auto i = 0; i < n_rows; ++i) {
    auto p0 = probs[i];
    auto p1 = probs[i + n_rows];
    output[i] = 1.0f * (p0 <= p1);
  }
}

void symTransform(const float *input, const param &params,
                  const program_t &final_progs, const int n_rows,
                  const int n_cols, float *output) {
  // cudaStream_t stream = handle.get_stream();
  // Execute final_progs(ordered by fitness) on input
  // output of size [n_rows,hall_of_fame]
  execute(final_progs, n_rows, params.n_components, input, output);
}

} // namespace genetic
