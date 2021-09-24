#!/usr/bin/env python
# coding: utf-8

# ## Chapter 01: Log-Linear Model
# 
# 
# ### 1. Introduction
# 
# Solomon and Wynne conducted an experiment on Dogs in 1953 . They wanted to understand, whether Dogs can learn from mistakes, so to speak. Specifically, They were interested in _avoidance-learning_. That is, when Dogs are given trauma-inducing shocks, will they learn to avoid shocks in future?
# 
# We can state the objectives of the expeirment, according to our understanding, in more general terms as follows:
# 
# 1. Can the Dogs learn?
# 2. Can they retain & recollect what they learnt?
# 
# The experimental setup, to drive the objectives, holds a dog in a closed compartment with steel flooring, open on one side with a small barrier for dog to jump over to the other side. A high-voltage electric shock is discharged into the steel floor intermittently to stimulate the dog. The dog is then left with an option to either get the shock for that trial or jump over the barrier to other side & save himself. Several dogs were recruited in the experiment.
# 
# The following picture ([source](https://commons.wikimedia.org/wiki/File:Shuttle_Box_Dog_Orange.png)) is an illustration of the setup.
# <br>
# <br>
# ![dog_setup](./data/avoidance_learning.png)
# <br>
# <br>
# More details of the experiment can be found [here](http://www.appstate.edu/~steelekm/classes/psy5300/Documents/Solomon&Wynne%201953.pdf).
# 
# <br>
# <br>

# In this chapter, we will analyze the experimental data using _Bayesian Analysis_, and the inference will be carried out in `Pyro`. The organization of the notebook is inspired from [Bayesian Workflow](http://www.stat.columbia.edu/~gelman/research/unpublished/Bayesian_Workflow_article.pdf) by Prof. Andrew Gelman et al. Another piece of work in that direction is from Betancourt et al [here](https://arxiv.org/abs/1904.12765). However, the current analysis is a WIP and far from perfect.
# 
# 
# An almost always first step in Bayesian Analysis is to elicit a plausible generative model, that would have likely generated the observed data. In this case, consider the model suggesed/implemented in WinBUGs [Vol1](https://www.mrc-bsu.cam.ac.uk/wp-content/uploads/WinBUGS_Vol1.pdf).
# 
# We want to model the relationship between __avoidance-in-future__ and __past-traumatic-experiences__ . The following log-linear model is a starting point:
# <br>
# <br>
# $\pi_j   =   A^{xj} B^{j-xj} $
# <br>
# <br>
# where :
#    * $\pi_j$ is the probability of a dog getting shocked at trial $j$
#    * $x_j$ is number of successful avoidances of shock prior to trial $j$.
#    * $j-x_j$ is number of shocks experienced prior to trial $j$.
#    * A & B, both are unknown and treated as random variables.
# 
# However, the model is only partially complete. In a Bayesian setting, we need to elicit our prior beliefs about the unknowns. Consequently, we need to give priors to $A$ and $B$, which we do shortly. Before that, we need some boiler plate code, mostly `imports`. Note that, all the code (functions) are glued in the [base](https://github.com/mlsquare/p3/blob/p3_part_2_ch01/Part_II/Chapter_1/chapter01.py) class. If one is interested, they can always browse the code [repo](https://github.com/mlsquare/p3) to get better understanding.

# In[1]:


import torch
import pyro
import pandas as pd
import itertools
from collections import defaultdict
import matplotlib.pyplot as plt
import pyro.distributions as dist
import seaborn as sns
import plotly
import plotly.express as px
import plotly.figure_factory as ff
from chapter01 import base
import numpy as np

pyro.set_rng_seed(1)

plt.style.use('default')

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')


# #### Data
# <br>
# The data contains experiments over 30 dogs and each dog is subjected to 25 trials.
# 
# The plot that follows highlights Dogs data in dictionary format, averaged over dog population for each trial, i.e., $y_j   =   \frac{1}{30}\sum{y_{ij}}$ where $y_{ij}$ is the i-th Dog's response at the j-th trial.

# In[2]:


dogs_data = base.load_data()
base.plot_original_y(1-np.mean(dogs_data["Y"], axis=0), ylabel='Probability of shock at trial j')


# Its apparent from experimental data that more than half the dog population learns to avoid shocks in trials as few as 5, and also that the learning doesn't significantly rise with increased number of trials.

# #### Preprocessing
# <br>
# Next, we need to transform raw data to obtain
# 
# - `x_avoidance` :  number of shock avoidances before current trial.
# - `x_shocked` :  number of shocks before current trial.
# - `y`: Status 'shocked (y=1) or avoided(y=0)' at current trial.
# 
# Here `pystan` format [data](https://github.com/stan-dev/example-models/blob/master/bugs_examples/vol1/dogs/dogs.data.R) (python dictionary) is passed to the function above, in order to preprocess it to tensor format required for pyro sampling.

# In[3]:


x_avoidance, x_shocked, y = base.transform_data(**dogs_data)
print("x_avoidance: %s, x_shocked: %s, y: %s"%(x_avoidance.shape, x_shocked.shape, y.shape))
print("\nSample x_avoidance: %s \n\nSample x_shocked: %s"%(x_avoidance[1], x_shocked[1]))

base.plot_original_y(x_avoidance.numpy(), ylabel='Cumulative Avoidances')
base.plot_original_y(x_shocked.numpy(), ylabel='Cumulative Shocked Trials')


# The original data is not revealing much, looking at the cumulative avoidances and shocks, we see that some dogs never learn (example: Dog 1-4), and some dogs learn and retain the learning behaviour (example: Dog 25-30).

# ### 2. Model Specification
# ________
# The sampling distrution of the generative model, as indicated earlier, is:
# <br>
# <br>
# $y_{ij}   \sim   Bern(\pi_{ij})$
# <br>
# $\log(\pi_{ij})   =   \alpha x_{ij} + \beta({j-x_{ij}})$
# <br>
# <br>
# Here, $y_{ij}=1$ if the a dog fails to avoid a shock at the j-th trial, and is 0 if it avoids. 
# 
# The above expression is used as a generalised linear model with log-link function in WinBugs implementation
# <br>
# #### BUGS model
# <br>
# $\log\pi_j = \alpha\ x_j + \beta\ ( $j$-x_j )$
#    
# Here
#    - $\log\pi_j$ is log probability of a dog getting shocked at trial $j$
#    - $x_j$ is number of successful avoidances of shock prior to trial $j$.
#    - $j-x_j$ is number of shocks experienced prior to trial $j$.
#    -  $\alpha$ is the coefficient corresponding to number of successes, $\beta$ is the coefficient corresponding to number of failures.
#   
# **Following code block is from original BUGS volume:**
# 
# 
# 
# ```model
#     {
#         for (i in 1 : Dogs) {
#         xa[i, 1] <- 0; xs[i, 1] <- 0 p[i, 1] <- 0
#         for (j in 2 : Trials) {
#         xa[i, j] <- sum(Y[i, 1 : j - 1])
#         xs[i, j] <- j - 1 - xa[i, j]
#         log(p[i, j]) <- alpha * xa[i, j] + beta * xs[i, j]
#         y[i, j] <- 1 - Y[i, j]
#         y[i, j] ~ dbern(p[i, j])
#        }
#     }
#     alpha ~ dnorm(0, 0.00001)I(, -0.00001)
#     beta ~ dnorm(0, 0.00001)I(, -0.00001)
#     A <- exp(alpha)
#     B <- exp(beta)
#     }
# ```
# 
# ____________________
# The same model when implemented in PyStan
#   
# #### Equivalent Stan model
# 
# ```
# {
#     alpha ~ normal(0.0, 316.2);
#     beta  ~ normal(0.0, 316.2);
#     for(dog in 1:Ndogs)
#         for (trial in 2:Ntrials)  
#             y[dog, trial] ~ bernoulli(exp(alpha * xa[dog, trial] + beta * xs[dog, trial]));
#       
# }  
# ```
# 
# 
# In the orginal WinBUGS reference, $\alpha, \beta$ are given truncated Normal priors, but not `Stan`. Note that $\pi_{ij} > 1$ when $\alpha, \beta > 0$. Thefore, we have to restrict  $\alpha, \beta \le 0$
# 
# Consequently, we elicit half-normal priors for $\alpha,\beta$ with zero mean and large variance (flat) to complete the specification. Also notice that, the above model is in a more familiar form (Generalized Linear Model or Log-Linear Model). For convenience, we can define $X_{a}\equiv x_{ij}, X_{s}\equiv j-x_{ij} $
# <br>
# <br>
# The complete model is:
# <br>
# <br>
# $y_{ij} \sim Bern(\pi_{ij})$ 
# <br>
# $\log(\pi_{ij})  =   \alpha X_{a} + \beta X_{s}$
# or
# $\pi_{ij}  =   \exp{(\alpha X_{a} + \beta X_{s})}$
# <br>
# $\alpha \sim N(0., 316.)I(\alpha <0)$
# <br>
# $\beta \sim N(0., 316.)I(\beta <0)$
# 
# But it is easier to constrain r.vs on the posive real line. So, we simply absorb the negative sign inside the model.  Finally, we have these models, with Half-normal and Uniform priors s.t $\alpha, \beta > 0$
# 
# We experiment with the variants of the above models as follows:
# 
# __Model A.__
# <br>
# $y_{ij} \sim Bern(\pi_{ij})$
# <br>
# $\pi_{ij}  =   \exp{(-(\alpha X_{a} + \beta X_{s}))}$
# 
# $\alpha \sim N(0., 316.)\ I(\alpha>0)$
# <br>
# $\beta \sim N(0., 316.)\ I(\beta>0)$
# 
# __Model B.__
# <br>
# $y_{ij} \sim Bern(\pi_{ij})$
# <br>
# $\pi_{ij}  =   \exp{(-(\alpha X_{a} + \beta X_{s}))}$
# 
# $\alpha \sim U(0, 10.)$
# <br>
# $\beta \sim U(0, 10.)$
# 
# 
# 
# 
# 

# #### Model implementation
# 
# The above model are defined in `base.DogsModel`

# In[4]:


DogsModel= base.DogsModel

DogsModel


# Let us also draw few samples from the prior, and look at the distribution

# In[5]:


num_samples = 1100 

alpha_prior_a, beta_prior_a= base.init_priors({"default":dist.HalfNormal(316.)})# Pass priors for model A
prior_samples_a = base.get_prior_samples(alpha_prior_a, beta_prior_a, num_samples=num_samples)

alpha_prior_b, beta_prior_b= base.init_priors({"default":dist.Uniform(0.001, 10)})# Pass uniform priors for model B
prior_samples_b = base.get_prior_samples(alpha_prior_b, beta_prior_b, num_samples=num_samples)


# Sampled output of prior values for alpha & beta is stored in `prior_samples` above, and is plotted on a KDE plot as follows:

# In[6]:



base.plot_prior_distributions(model_halfnormal_a= prior_samples_a, model_uniform_b= prior_samples_b)


# ### 3. Prior predictive checking

# In[7]:



original_plus_simulated_prior_data = base.simulate_observations_given_prior_posterior_pairs(dogs_data["Y"], num_dogs=30, num_trials=24, 
                                                                                            activation_type= "exp", model_halfnormal_a= prior_samples_a, 
                                                                                            model_uniform_b= prior_samples_b)


# Even though the mean and median can differ from run to run, looking at the densities, the variance is vary large -- so the sample mean would also have huge variance. The take-away is that, both $\alpha, \beta$ are given very weak priors. We more or less rely on the data (evidence) to drive their estimation.
# 
# 
# TBD: Prior Sensitivity Analysis

# ### 4. Posterior Estimation
# 
# In the Bayesian setting, inference is drawn from the posterior.  Here, posterior implies the updated beliefs about the random variables, in the wake of given evidences (data). Formally,
# <br>
# <br>
# $Posterior = \frac {Likelihood x Prior}{Probability \ of Evidence}$
# <br>
# <br>
# In our case, $\alpha,\beta$ are the parameters (actually random variables) & $y$ is the evidence; Posterior $P(\alpha,\beta | y)$ is given, according to the Bayes rule, as:
# <br>
# <br>
# $P\ (\alpha,\beta | y) = \frac {P(y | \alpha,\beta) \pi(\alpha,\beta)}{P(y)}$
# <br>
# <br>
# 
# Now our interest is in estimating the posterior summaries of the parameters $\alpha, \beta$. For example, we can look at the posterior of mean of $\alpha$, denoted as $E(\alpha)$. However, in order to the get the posterior quanitities, either we need to compute the integrals or approximate the integrals via Markov Chain Monte Carlo. 
# 
# The latter can be easily accomplished in Pyro by using the NUTS sampler -- NUTS is a specific sampler designed to draw samples efficiently from the posterior using Hamiltonian Monte Carlo dynamics.
# 
# The following code snippet takes a [pyro](http://pyro.ai/examples/intro_part_i.html) model object with posterior specification, input data, some configuration parameters such as a number of chains and number of samples per chain. It then laucnhes a [NUTS](https://arxiv.org/pdf/1111.4246.pdf) sampler and produces MCMC samples in a python dictionary format.

# In[8]:


# DogsModel_A alpha, beta ~ HalfNormal(316.) & activation "exp"

hmc_sample_chains_a, hmc_chain_diagnostics_a = base.get_hmc_n_chains(DogsModel, x_avoidance, x_shocked, y, num_chains=4, 
                                                                     sample_count = 900, alpha_prior= dist.HalfNormal(316.), 
                                                                     beta_prior= dist.HalfNormal(316.), activation= "exp")


# In[9]:


# DogsModel_B alpha, beta ~ Uniform(0., 10.0) & activation "exp"

hmc_sample_chains_b, hmc_chain_diagnostics_b = base.get_hmc_n_chains(DogsModel, x_avoidance, x_shocked, y, num_chains=4, sample_count = 900, 
                         alpha_prior= dist.Uniform(0., 10.0), beta_prior= dist.Uniform(0., 10.0), activation= "exp")


# `hmc_sample_chains` holds sampled MCMC values as `{"Chain_0": {alpha	[-0.20020795, -0.1829252, -0.18054989 . .,], "beta": {}. .,}, "Chain_1": {alpha	[-0.20020795, -0.1829252, -0.18054989 . .,], "beta": {}. .,}. .}`

# ### 5. Diagnosing the computational approximation
# 
# Just like any numerical technique, no matter how good the theory is or how robust the implementation is, it is always a good idea to check if indeed the samples drawn are reasonable. In the ideal situation, we expect the samples drawm by the sampler to be independant, and identically distributed (i.i.d) as the posterior distribution. In practice, this is far from true as MCMC itself is an approxmate technique and a lot can go wrong. In particular, chains may not have converged or samples are very correlated.
# 
# We can use both visual and more formal statistical techniques to inspect the quality of the fit (not the model fit to the data, but how well the appximation is, having accepted the model class for the data at hand) by  treating chains as time-series data, and that we can run several chains in parallel. We precisely do that next. 
# <br>
# <br>
# Following snippet allows plotting **Parameter vs. Chain matrix** and optionally saving the dataframe.

# In[10]:


# Unpruned sample chains for model A with HalfNormal prior

beta_chain_matrix_df_A = pd.DataFrame(hmc_sample_chains_a)

base.save_parameter_chain_dataframe(beta_chain_matrix_df_A, "data/dogs_parameter_chain_matrix_1A.csv")


# In[11]:


# Unpruned sample chains for model B with Uniform prior

beta_chain_matrix_df_B = pd.DataFrame(hmc_sample_chains_b)

base.save_parameter_chain_dataframe(beta_chain_matrix_df_B, "data/dogs_parameter_chain_matrix_1B.csv")


# #### Sample chains mixing for `HalfNormal` & `Uniform` priors
# Following plots chains of samples for alpha & beta parameters

# In[12]:


print("For 'model_HalfNormal_a'")    
base.plot_chains(beta_chain_matrix_df_A)


print("\nFor 'model_Uniform_b'")    
base.plot_chains(beta_chain_matrix_df_B)


# In[13]:


for chain, samples in hmc_sample_chains_a.items():
    print("____\nFor model_HalfNormal_a %s"%chain)    
    samples= dict(map(lambda param: (param, torch.tensor(samples.get(param))), samples.keys()))# np array to tensors
    print(chain, "Sample count: ", len(samples["alpha"]))
    print("Alpha Q(0.5) :%s | Beta Q(0.5) :%s"%(torch.quantile(samples["alpha"], 0.5), torch.quantile(samples["beta"], 0.5)))

print("%s\n%s"%("_"*60, "_"*60))

for chain, samples in hmc_sample_chains_b.items():
    print("____\nFor model_Uniform_b %s"%chain)    
    samples= dict(map(lambda param: (param, torch.tensor(samples.get(param))), samples.keys()))# np array to tensors
    print(chain, "Sample count: ", len(samples["alpha"]))
    print("Alpha Q(0.5) :%s | Beta Q(0.5) :%s"%(torch.quantile(samples["alpha"], 0.5), torch.quantile(samples["beta"], 0.5)))


# In[14]:


for chain in hmc_sample_chains_a.keys():
    print("____\nFor 'model_HalfNormal_a' & 'model_Uniform_b' %s"%chain)
    chain_list= list(hmc_sample_chains_a[chain].values())
    chain_list.extend(list(hmc_sample_chains_b[chain].values()))
    title= "parameter distribution for : %s"%(chain)
    fig = ff.create_distplot(chain_list, ["alpha_a", "beta_a", "alpha_b", "beta_b"])
    fig.update_layout(title=title, xaxis_title="parameter values", yaxis_title="density", legend_title="parameters")
    fig.show()
    break


# #### Auto-correlation plots for sample chains with `HalfNormal` priors

# In[15]:



base.autocorrelation_plots(beta_chain_matrix_df_A)


# #### Auto-correlation plots for sample chains with `Uniform` priors

# In[16]:



base.autocorrelation_plots(beta_chain_matrix_df_B)


# For `alpha`, `thining factor` for `chain_0` is 3, `chain_1` is 3, `chain_2` is 3, `chain_3` is 3
# 
# for `beta`, `thinin factor` for `chain_0` is 3, `chain_1` is 3, `chain_2` is 3, `chain_3` is 3
# 
# 

# #### Pruning chains from model with `HalfNormal` priors

# In[17]:


thining_dict_a = {"chain_0": {"alpha":3, "beta":3}, "chain_1": {"alpha":3, "beta":3}, 
                "chain_2": {"alpha":3, "beta":3}, "chain_3": {"alpha":3, "beta":3}}

pruned_hmc_sample_chains_a = base.prune_hmc_samples(hmc_sample_chains_a, thining_dict_a)


# #### Pruning chains from model with `Uniform` priors

# In[18]:


thining_dict_b = {"chain_0": {"alpha":3, "beta":3}, "chain_1": {"alpha":3, "beta":3}, 
                "chain_2": {"alpha":3, "beta":3}, "chain_3": {"alpha":3, "beta":3}}

pruned_hmc_sample_chains_b = base.prune_hmc_samples(hmc_sample_chains_b, thining_dict_b)


# #### Gelman-Rubin statistic for chains with `HalfNormal` priors

# In[19]:



grubin_values_a = base.gelman_rubin_stats(pruned_hmc_sample_chains_a)


# #### Gelman-Rubin statistic for chains with `Uniform` priors

# In[20]:



grubin_values_b = base.gelman_rubin_stats(pruned_hmc_sample_chains_b)


# In[21]:


# Following diagnostic table is for Un-pruned chains from sampler with Normal prior.

base.get_chain_diagnostics(hmc_chain_diagnostics_a)


# In[22]:


# Following diagnostic table is for Un-pruned chains from sampler with Uniform prior.

base.get_chain_diagnostics(hmc_chain_diagnostics_b)


# Based on simple multiple line plots, we can see that, in this run, `chain_3` is behaving differently than the remaining chains. It may be due to really a different initialization. Otherwise, all chains seem to mix well. Therefore, we drop `chain_3` from analysis. However, we need to be cautious about dropping, and we should check what is its effect on the actual predictions -- since sometimes, even though parameters can look very different numerically, they may have very little effect on the likelihood. Neverthless, it implies that either something is not right about the chain or the model is operating at the edge.
# 
# 
# #### Descriptive summaries
# 
# Following outputs the summary of required statistics such as `"mean", "std", "Q(0.25)", "Q(0.50)", "Q(0.75)"`, select names of statistic metric from given list to view values

# #### Summary for model chains with `HalfNormal` priors

# In[23]:


#chain results Pruned after ACF plots

beta_chain_matrix_df_A = pd.DataFrame(pruned_hmc_sample_chains_a)

base.summary(beta_chain_matrix_df_A)


# #### Summary for model chains with `Uniform` priors

# In[24]:


#chain results Pruned after ACF plots

beta_chain_matrix_df_B = pd.DataFrame(pruned_hmc_sample_chains_b)

base.summary(beta_chain_matrix_df_B)


# We can also report the 5-point Summary Statistics (mean, Q1-Q4, Std, ) as tabular data per chain and save the dataframe

# #### 5-point Summary Statistics for model chains with `HalfNormal` priors

# In[25]:


fit_df_A = pd.DataFrame()
for chain, values in pruned_hmc_sample_chains_a.items():
    param_df = pd.DataFrame(values)
    param_df["chain"]= chain
    fit_df_A= pd.concat([fit_df_A, param_df], axis=0)

base.save_parameter_chain_dataframe(fit_df_A, "data/dogs_classification_hmc_samples_1A.csv")


# #### 5-point Summary Statistics for model chains with `Uniform` priors

# In[26]:


fit_df_B = pd.DataFrame()
for chain, values in pruned_hmc_sample_chains_b.items():
    param_df = pd.DataFrame(values)
    param_df["chain"]= chain
    fit_df_B= pd.concat([fit_df_B, param_df], axis=0)

base.save_parameter_chain_dataframe(fit_df_B, "data/dogs_classification_hmc_samples_1B.csv")


# Use following button to upload:
# 
# * `"data/dogs_classification_hmc_samples_1A.csv"` as `'fit_df_A'`
# 
# * `"data/dogs_classification_hmc_samples_1B.csv"` as `'fit_df_B'`

# In[27]:


# Use following to load data once the results from pyro sampling operation are saved offline
load_button= base.build_upload_button()


# In[28]:


# Use following to load data once the results from pyro sampling operation are saved offline

if load_button.value:
    fit_df_A = base.load_parameter_chain_dataframe(load_button)#Load "data/dogs_classification_hmc_samples_1A.csv"
#     fit_df_B= base.load_parameter_chain_dataframe(load_button)#Load "data/dogs_classification_hmc_samples_1B.csv"


# Following outputs the similar summary of required statistics such as `"mean", "std", "Q(0.25)", "Q(0.50)", "Q(0.75)"`, but in a slightly different format, given a list of statistic names

# #### Summary for model chains with `HalfNormal` priors

# In[29]:



base.summary(fit_df_A, layout =2)


# #### Summary for model chains with `Uniform` priors

# In[30]:



base.summary(fit_df_B, layout =2)


# Following plots sampled parameters values as Boxplots with `M parameters` side by side on x axis for each of the `N chains`

# Pass the list of `M parameters` and list of `N chains`, with `plot_interactive` as `True or False` to choose between _Plotly_ or _Seaborn_

# In[31]:


parameters= ["alpha", "beta"]# All parameters for given model
chains= fit_df_A["chain"].unique()# Number of chains sampled for given model


# #### Boxplot for model chains with `HalfNormal` priors

# In[32]:


# Use plot_interactive=False for Normal seaborn plots offline

base.plot_parameters_for_n_chains(fit_df_A, chains=['chain_0', 'chain_1', 'chain_2', 'chain_3'], parameters=parameters, plot_interactive=True)


# #### Boxplot for model chains with `Uniform` priors

# In[33]:


# Use plot_interactive=False for Normal seaborn plots offline

base.plot_parameters_for_n_chains(fit_df_B, chains=['chain_0', 'chain_1', 'chain_2', 'chain_3'], parameters=parameters, plot_interactive=True)


# Following plots the `joint distribution` of `pair of each parameter` sampled values for all chains with `HalfNormal` priors.

# In[34]:



base.plot_joint_distribution(fit_df_A, parameters)


# Following plots the `joint distribution` of `pair of each parameter` sampled values for all chains with `Uniform` priors.

# In[35]:



base.plot_joint_distribution(fit_df_B, parameters)


# Following plots the `Pairplot distribution` of each parameter with every other parameter's sampled values for all chains with `HalfNormal` priors

# In[36]:


sns.pairplot(data=fit_df_A, hue= "chain");


# Following plots the `Pairplot distribution` of each parameter with every other parameter's sampled values for all chains with `Uniform` priors

# In[37]:


sns.pairplot(data=fit_df_B, hue= "chain");


# Based on all above summaries both visual and descriptive, `chain_2` seemed problematics, and it is very clear that $\alpha, \beta <0$ with almost certainly.
# 
# TBD: Converence Statsitics like Gelman-Rubin has to be implemented.

# ### 6. Sensitivity Analysis
# 
# Posterior Predictive Checking (PPE) helps examine the fit of a model to real data, as the parameter drawn for simulating conditions & regions of interests come from the posterior distribution. While PPE incorporates model uncertainly (by averaring over all possible models), we take a simpler route to begin with, which is to, sample the $\alpha, \beta$ pair that is very plausible in the posterior (eg. the poster means), and simulate data from under this particular generative model.

# In[38]:


# With priors & posteriros
posterior_parameters_pairs_a = pruned_hmc_sample_chains_a.get('chain_0')

posterior_parameters_pairs_b = pruned_hmc_sample_chains_b.get('chain_0')

original_plus_simulated_data_posterior = base.simulate_observations_given_prior_posterior_pairs(dogs_data["Y"], 
                                                                                                num_dogs=30, num_trials=24, activation_type= "exp",
                                                                                                prior_simulations= original_plus_simulated_prior_data, 
                                                                                                model_halfnormal_a= posterior_parameters_pairs_a,
                                                                                                model_uniform_b= posterior_parameters_pairs_b)


# Save & Export simulated data from `prior & posterior predictive checking` to _chapter 02_

# In[47]:


original_plus_simulated_data_posterior_df= pd.DataFrame(original_plus_simulated_data_posterior.T, 
                                                     columns=["Dog_%s"%(idx+1) for idx in range(len(original_plus_simulated_data_posterior))])


base.save_parameter_chain_dataframe(original_plus_simulated_data_posterior_df, 
                                    "data/dogs_original_plus_simulated_data_model_1ab.csv")


# ### 7. Model Comparison
# 
# More often than not, there may be many plausible models that can explain the data. Sometime, the modeling choice is based on _domain knowledge_. Sometime it is out of comptational conveninece. Latter is the case with the choice of priors. One way to consider different models is by eliciting different prior distributions. 
# 
# As long as the sampling distribtion is same, one can use Deviance Information Criterion (DIC) to guide model comparison.

# #### Deviance Information Criterion
# 
# DIC is computed as follows
# <Br>
# 
# $D(\alpha,\beta) = -2\ \sum_{i=1}^{n} \log P\ (y_{i}\ /\ \alpha,\beta)$
# 
# $\log P\ (y_{i}\ /\ \alpha,\beta)$ is the log likehood of shocks/avoidances observed given parameter $\alpha,\beta$, this expression expands as follows:
# 
# $$D(\alpha,\beta) = -2\ \sum_{i=1}^{30}[ y_{i}\ (\alpha Xa_{i}\ +\beta\ Xs_{i}) + \ (1-y_{i})\log\ (1\ -\ e^{(\alpha Xa_{i}\ +\beta\ Xs_{i})})]$$
# 
# <Br>
# Using $D(\alpha,\beta)$ to Compute DIC
# <Br>
# 
# $\overline D(\alpha,\beta) = \frac{1}{T} \sum_{t=1}^{T} D(\alpha,\beta)$
# 
# $\overline \alpha = \frac{1}{T} \sum_{t=1}^{T}\alpha_{t}\\$
# $\overline \beta = \frac{1}{T} \sum_{t=1}^{T}\beta_{t}$
# 
# $D(\overline\alpha,\overline\beta) = -2\ \sum_{i=1}^{30}[ y_{i}\ (\overline\alpha Xa_{i}\ +\overline\beta\ Xs_{i}) + \ (1-y_{i})\log\ (1\ -\ e^{(\overline\alpha Xa_{i}\ +\overline\beta\ Xs_{i})})]$
# 
# <Br>
# Therefore finally
# <Br>
# 
# $DIC\ =\ 2\ \overline D(\alpha,\beta)\ -\ D(\overline\alpha,\overline\beta)$
# 
# <Br>
# <Br>

# <Br>
# Following method computes deviance value given parameters `alpha & beta`
# <Br>

# In[36]:


#launch docstring for calculate_deviance_given_param

#launch docstring for calculate_mean_deviance


# <Br>
# 
# Following method computes `deviance information criterion` for a given bayesian model & chains of sampled parameters `alpha & beta`
# 
# <Br>

# In[37]:


#launch docstring for DIC

#launch docstring for compare_DICs_given_model


# Sample an alternate model with different priors
# 
# The alternate model is defined in the same manner using Pyro as per the following expression of generative model for this dataset, just with modification of prior distribution to `Uniform` rather than `Normal` as follows:
# 
# Instead of considering Normal priors of $\alpha$ and $\beta$, we consider uniform priors, i.e.,
# $prior\ \alpha$ ~ $U(0, 10.)$,  $\beta$ ~ $U(0, 10.)$
# 
# In _Section 5._, we already sampled such model to obtain posterior chains as `pruned_hmc_sample_chains_b` which we shall use for computing DIC in cells that follow.

# #### Sample chains mixing for Uniform priors
# Following plots chains of samples for alpha & beta parameters with uniform priors

# In[60]:



base.plot_chains(pd.DataFrame(pruned_hmc_sample_chains_b))


# #### Gelman-Rubin statistic for Uniform priors

# In[49]:



grubin_values_uniform = base.gelman_rubin_stats(pruned_hmc_sample_chains_b)


# #### compute & compare `deviance information criterion` for a multiple bayesian models

# In[50]:



base.compare_DICs_given_model(x_avoidance, x_shocked, y, Dogs_HalfNormal_prior= pruned_hmc_sample_chains_a, Dogs_uniform_prior= pruned_hmc_sample_chains_b)


# The DIC values are very close, so we don't anticipate substantially different fits. This is largely because, both priors are flat. However, if were to follow the rule book, we had to pick a model with the smallest DIC. In this case, both `Normal` Priors & `Uniform` Priors yield identical DIC.

# ### 8. Inference & Analysis
# 
# Alright, we have a model, and we are reasonable sure about the fit (both numerical and conceptual), but so what? The purpose of model building is to use these models as probing devices. That is, using the models can we answer some questions about the reality that these models have abstracted. 
# 
# 
# We choose model with Normal Prior, and pick samples from one particular chain of HMC samples say `chain_3`

# In[53]:


for chain, samples in pruned_hmc_sample_chains_a.items():
    samples= dict(map(lambda param: (param, torch.tensor(samples.get(param))), samples.keys()))# np array to tensors
    print(chain, "Sample count: ", len(samples["alpha"]))


# Plot density for parameters from `chain_3` to visualise the spread of sample values from that chain

# In[54]:


title= "parameter distribution for : %s"%(chain)
fig = ff.create_distplot(list(map(lambda x:x.numpy(), samples.values())), list(samples.keys()))
fig.update_layout(title=title, xaxis_title="parameter values", yaxis_title="density", legend_title="parameters")
fig.show()

print("Alpha Q(0.5) :%s | Beta Q(0.5) :%s"%(torch.quantile(samples["alpha"], 0.5), torch.quantile(samples["beta"], 0.5)))


# Plot density & contours for both parameters from `chain_3` to visualise the joint distribution & region of interest with `HalfNormal` priors.

# In[55]:


#Choosing samples from chain 3
chain_samples_df_A= fit_df_A[fit_df_A["chain"]==chain].copy()# chain is 'chain_3' 

alpha= chain_samples_df_A["alpha"].tolist()
beta= chain_samples_df_A["beta"].tolist()
colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)', (1, 1, 0.2), (0.98,0.98,0.98)]
fig = ff.create_2d_density(alpha, beta, colorscale=colorscale, hist_color='rgb(255, 255, 150)', point_size=4, title= "Alpha-Beta joint density plot for model with 'HalfNormal' prior")
fig.update_layout( xaxis_title="x (alpha)", yaxis_title="y (beta)")

fig.show()


# **Note:** The distribution of alpha values are significantly offset to the left from beta values, by almost 26 times; Thus for any given input observation of avoidances or shocks, the likelihood of getting shocked is more influenced by small measure of avoidance than by getting shocked.

# #### Observations:
# 
# On observing the joint distribution of $\alpha, \beta$, we note that $\beta > \alpha$  and $\beta$ is closer to zero. 
# Here, $\beta$ can be interpreted as _learning ability_, i.e., the ability of a dog to learn from _shock_ experiences. The increase in number of shocks barely raises the probability of non-avoidance (value of 𝜋𝑗) with little amount. Unless the trials & shocks increase considerably large in progression, it doesn't mellow down well and mostly stays around 0.9.
# 
# However, it is not the case with $\alpha, \alpha$ is more negative & farthest from 'zero'. It imparts a significant decline in non-avoidance (𝜋𝑗) even for few instances where dog avoids the shock; therefore $\alpha$ can be interpreted as _retention ability_ i.e., the ability to retain the learning from previous shock experiences.

# In[56]:


print(chain_samples_df_A["alpha"].describe(),"\n\n", chain_samples_df_A["beta"].describe())


# From the contour plot above following region in posterior distribution seems highly plausible for parameters:
# 1. For alpha, `0.18 < alpha < -0.20`
# 2. For beta `0.005 < beta < 0.008`

# Let us look at $\frac{\alpha}{\beta}$ as a proxy to see which of the two (_learning ability_ and _retention_ability) are domimant. 
# 
# We are using $\frac{\alpha}{\beta}$ as a probing device to answer that question, and similar quantities can be defined. With MCMC samples available, we can get posterior probabilties of any function of the model parameters (here $\alpha, \beta$. Say, we can be interested in the $E(\frac{\alpha}{\beta})$ or $P(\frac{\alpha}{\beta}<1)$.
# 
# The latter quantity can be estimate by the Monte Carlo average as follows:
# $P(\frac{\alpha}{\beta}>1) = \frac{1}{n}\sum_{t=1}^{n} I(\alpha < \beta)$, i.e, the fraction of times $\alpha < \beta$.

# In[59]:


x1 = chain_samples_df_A["alpha"].to_numpy()
x2 = chain_samples_df_A["beta"].to_numpy()
p = np.mean(x1<x2)
print(p)


# So, the posterior evident for _retention ability_ outweigting _learning abilty_ is overwhelming.  

# ### 9. Commentary
# .
# 
# TBD.

# ____________________
