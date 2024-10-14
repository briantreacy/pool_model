import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

class Parameters_SEIR_pool_testing:
    
    def __init__(self, R0, Npool, r_v, t_E, t_I, t_P, t_S, t_Q, t_linspace):
        
        self.R0 = R0
        self.Npool = Npool
        self.r_v = r_v
        
        self.t_E = t_E
        self.sigma = 1 / t_E
        
        self.t_I = t_I
        self.gamma = 1 / t_I
        
        self.t_P = t_P
        self.Omega = 1 / t_P
        
        self.t_S = t_S
        self.omega = 1 / t_S
        
        self.t_Q = t_Q
        self.delta = 1 / t_Q
        
        self.beta = R0 * self.gamma
        self.t_linspace = t_linspace
        
        if r_v * t_P >= 1:
            
            print("r_v * t_P must be less than Npool")
            
        else:
            
            self.r_p = r_v / (1  - r_v * t_P)
    
class SEIR_pool_testing:
    
    def __init__(self, param_obj):
        
        self.beta = param_obj.beta
        self.Npool = param_obj.Npool
        self.r_v = param_obj.r_v
        self.sigma = param_obj.sigma
        self.gamma = param_obj.gamma
        self.Omega = param_obj.Omega
        self.omega = param_obj.omega
        self.delta = param_obj.delta
        self.t_linspace = param_obj.t_linspace
        self.r_p = param_obj.r_p
        self.sol = self.generate_solution().y
        self.R0 = param_obj.R0
        self.R0_p = self.generate_R0_p(param_obj)
           
    def prob_positive_pool_negative(self, x):
        
        # This returns the probability of an individual who isn't infected themselves of  being in a pool test
        # with Npool - 1 other individuals that returns a negative result given the prevalence is x
        # Note the prevalance x is really the prevalence of individuals that would trigger a positive
        # pool test rather than the prevalence of infection. The I compartment in this model
        
        return 1 - (1 - x) ** (self.Npool - 1)
    
    def prob_negative_pool_negative(self, x):
        
        # This returns the probability of an individual who isn't infected themselves of  being in a pool test
        # with Npool - 1 other individuals that returns a positive result given the prevalence is x
        # Note the prevalance x is really the prevalence of individuals that would trigger a positive
        # pool test rather than the prevalence of infection. The I compartment in this model
        
        return (1 - x) ** (self.Npool - 1)
        
    def fun(self, t, y):
        
        S, Sppos, Spneg, Sneg = y[0:4]
        E, Eppos, Epneg, Epos, EQ = y[4:9]
        I, Ippos, Ipneg, Ipos, IQ = y[9:14]
        R, Rppos, Rpneg, Rpos, Rneg, RQ = y[14:20]
        
        x = I/(S + E + I + R)
        
        d_S_dt = self.omega * Sneg + self.Npool * self.Omega * Spneg - self.beta * S * (I + Ippos + Ipneg) - self.Npool * self.r_p * S
        d_Sppos_dt = self.Npool * self.r_p * self.prob_positive_pool_negative(x) * S - self.beta * Sppos * (I + Ippos + Ipneg) - self.Npool * self.Omega * Sppos
        d_Spneg_dt = self.Npool * self.r_p * self.prob_negative_pool_negative(x) * S - self.beta * Spneg * (I + Ippos + Ipneg) - self.Npool * self.Omega * Spneg
        d_Sneg_dt = self.Npool * self.Omega * Sppos - self.omega * Sneg
        
        d_E_dt = self.beta * S * (I + Ippos + Ipneg) + self.Npool * self.Omega * Epneg + self.delta * EQ - self.Npool * self.r_p * E - self.sigma * E
        d_Eppos_dt = self.beta * Sppos * (I + Ippos + Ipneg) + self.Npool * self.prob_positive_pool_negative(x) * self.r_p * E - self.Npool * self.Omega * Eppos - self.sigma * Eppos
        d_Epneg_dt = self.beta * Spneg * (I + Ippos + Ipneg) + self.Npool * self.prob_negative_pool_negative(x) * self.r_p * E - self.Npool * self.Omega * Epneg - self.sigma * Epneg
        d_Epos_dt = self.Npool * self.Omega * Eppos - self.omega * Epos - self.sigma * Epos
        d_EQ_dt = self.omega * Epos - self.delta * EQ - self.sigma * EQ
        
        d_I_dt = self.sigma * E + self.Npool * self.Omega * Ipneg + self.delta * IQ - self.Npool * self.r_p * I - self.gamma * I
        d_Ippos_dt = self.sigma * Eppos + self.Npool * self.r_p * I - self.Npool * self.Omega * Ippos - self.gamma * Ippos
        d_Ipneg_dt = self.sigma * Epneg - self.Npool * self.Omega * Ipneg - self.gamma * Ipneg
        d_Ipos_dt = self.sigma * Epos + self.Npool * self.Omega * Ippos - self.omega * Ipos - self.gamma * Ipos
        d_IQ_dt = self.sigma * EQ + self.omega * Ipos - self.delta * IQ - self.gamma * IQ
        
        d_R_dt = self.gamma * I + self.Npool * self.Omega * Rpneg + self.omega * Rneg + self.delta * RQ - self.Npool * self.r_p * R
        d_Rppos_dt = self.gamma * Ippos + self.Npool * self.prob_positive_pool_negative(x) * self.r_p * R - self.Npool * self.Omega * Rppos
        d_Rpneg_dt = self.gamma * Ipneg + self.Npool * self.prob_negative_pool_negative(x) * self.r_p * R - self.Npool * self.Omega * Rpneg
        d_Rpos_dt = self.gamma * Ipos - self.omega * Rpos
        d_Rneg_dt = self.Npool * self.Omega * Rppos - self.omega * Rneg
        d_RQ_dt = self.gamma * IQ + self.omega * Rpos - self.delta * RQ
        
        dydt = [d_S_dt, d_Sppos_dt, d_Spneg_dt, d_Sneg_dt, d_E_dt, d_Eppos_dt, d_Epneg_dt, d_Epos_dt, d_EQ_dt, d_I_dt, d_Ippos_dt, d_Ipneg_dt, d_Ipos_dt, d_IQ_dt, d_R_dt, d_Rppos_dt, d_Rpneg_dt, d_Rpos_dt, d_Rneg_dt, d_RQ_dt]
        
        return dydt
    
    def generate_solution(self, y0 = None, method = 'LSODA'):
        
        # This generates a solution over a linear time space t_linspace
        
        if y0 == None:
            y0 = [7999999/8000000, 0, 0, 0, 1/8000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        t0, tf = self.t_linspace[0], self.t_linspace[-1]
        return solve_ivp(self.fun, (t0, tf), y0, method, t_eval = self.t_linspace)
    
    def S(self, compartment_string = None):
        
        if compartment_string == None:
            
            return self.sol[0]
        
        elif compartment_string == 'ppos':
            
            return self.sol[1]
        
        elif compartment_string == 'pneg':
            
            return self.sol[2]
        
        elif compartment_string == 'neg':
            
            return self.sol[3]
    
    def E(self, compartment_string = None):
        
        if compartment_string == None:
            
            return self.sol[4]
        
        elif compartment_string == 'ppos':
            
            return self.sol[5]
        
        elif compartment_string == 'pneg':
            
            return self.sol[6]
        
        elif compartment_string == 'pos':
            
            return self.sol[7]
        
        elif compartment_string == 'Q':
            
            return self.sol[8] 
        
    def I(self, compartment_string = None):
        
        if compartment_string == None:
            
            return self.sol[9]
        
        elif compartment_string == 'ppos':
            
            return self.sol[10]
        
        elif compartment_string == 'pneg':
            
            return self.sol[11]
        
        elif compartment_string == 'pos':
            
            return self.sol[12]
        
        elif compartment_string == 'Q':
            
            return self.sol[13]
        
    def R(self, compartment_string = None):
        
        if compartment_string == None:
            
            return self.sol[14]
        
        elif compartment_string == 'ppos':
            
            return self.sol[15]
        
        elif compartment_string == 'pneg':
            
            return self.sol[16]
        
        elif compartment_string == 'pos':
            
            return self.sol[17]
        
        elif compartment_string == 'neg':
            
            return self.sol[18]
        
        elif compartment_string == 'Q':
            
            return self.sol[19]
        
    def E_proportion_infected_population(self):
        
        return [E / (E + I) for (E, I) in zip(self.E(), self.I())]
    
    def I_proportion_infected_population(self):
        
        return [I / (E + I) for (E, I) in zip(self.E(), self.I())]        
        
    def sensitivity(self):
        
        # Returns the sensitivity of the pool tests over a linear time space.
        # Sensitivity is the probability that a pool test will return a positive
        # result given there is someone infected in the pool.
          
        return [((S + E + I + R) ** self.Npool - (S + E + R) ** self.Npool) / ((S + E + I + R) ** self.Npool - (S + R) ** self.Npool ) for (S, E, I, R) in zip(self.S(), self.E(), self.I(), self.R())]
    
    def r_SEIR_traditional(self):
        
        return (1/2) * (- (self.sigma + self.gamma) + np.sqrt((self.sigma + self.gamma)**2 + 4 * self.sigma * (self.beta - self.gamma)))  
    
    def E_proportion_infected_population_limiting_SEIR_traditional(self):
        
        r = self.r_SEIR_traditional()
        
        return (self.gamma + r) / (self.sigma + self.gamma + r)

    def I_proportion_infected_population_limiting_SEIR_traditional(self):
        
        r = self.r_SEIR_traditional()
        
        return (self.sigma) / (self.sigma + self.gamma + r)

    def prob_step(self, compartment_from, compartment_to, param_obj):
        
        if compartment_from == "E":
            
            if compartment_to == "Epneg":
                
                return param_obj.Npool * param_obj.r_p / (param_obj.Npool * param_obj.r_p + param_obj.sigma)
            
            elif compartment_to == "I":
                
                return param_obj.sigma / (param_obj.Npool * param_obj.r_p + param_obj.sigma)
            
            else:
                
                return 0
            
        elif compartment_from == "Epneg":
            
            if compartment_to == "E":
                
                return param_obj.Npool * param_obj.Omega / (param_obj.Npool * param_obj.Omega + param_obj.sigma)
            
            elif compartment_to == "Ipneg":
                
                return param_obj.sigma / (param_obj.Npool * param_obj.Omega + param_obj.sigma)
            
            else:
                
                return 0
            
        elif compartment_from == "I":
        
            if compartment_to == 'Ippos':
                
                return param_obj.Npool * param_obj.r_p / (param_obj.Npool * param_obj.r_p + param_obj.gamma)
            
            elif compartment_to == 'R':
                
                return param_obj.gamma / (param_obj.Npool * param_obj.r_p + param_obj.gamma)
            
            else:
                
                return 0
            
        elif compartment_from == "Ippos":
        
            if compartment_to == "Ipos":
                
                return param_obj.Npool * param_obj.Omega / (param_obj.Npool * param_obj.Omega + param_obj.gamma)
            
            elif compartment_to == "Rppos":
                
                return param_obj.gamma / (param_obj.Npool * param_obj.Omega + param_obj.gamma)
            
            else:
                
                return 0
        
        elif compartment_from == "Ipneg":
            
            if compartment_to == "I":
                
                return param_obj.Npool * param_obj.Omega / (param_obj.Npool * param_obj.Omega + param_obj.gamma)
            
            elif compartment_to == "Rpneg":
                
                return param_obj.gamma / (param_obj.Npool * param_obj.Omega + param_obj.gamma)
            
            else:
                
                return 0
            
        elif compartment_from == "Ipos":
            
            if compartment_to == "IQ":
                
                return param_obj.omega / (param_obj.omega + param_obj.gamma)
            
            elif compartment_to == "Rpos":
                
                return param_obj.gamma / (param_obj.omega + param_obj.gamma)
            
            else:
                
                return 0            
        
        elif compartment_from == "IQ":
            
            if compartment_to == "I":
                
                return param_obj.delta / (param_obj.delta + param_obj.gamma)
            
            elif compartment_to == "RQ":
                
                return param_obj.gamma / (param_obj.delta + param_obj.gamma)
            
            else:
                
                return 0
            
    def prob_chain(self, compartments, param_obj):
        
        chain_probability = 1
        
        for i in range(len(compartments)-1):
            
            chain_probability = chain_probability * self.prob_step(compartments[i], compartments[i + 1], param_obj)
            
        return chain_probability
    
    def prob_return(self, compartment, param_obj):
        
        if compartment == "E":
            
            return self.prob_chain(['E', 'Epneg', 'E'], param_obj)
        
        elif compartment == 'Epneg':
            
            return self.prob_chain(['Epneg', 'E', 'Epneg'], param_obj)
        
        elif compartment == 'I':
            
            return self.prob_chain(['I', 'Ippos', 'Ipos', 'IQ', 'I'], param_obj)
        
        elif compartment == 'Ippos':
            
            return self.prob_chain(['Ippos', 'Ipos', 'IQ', 'I', 'Ippos'], param_obj)   
        
        elif compartment == 'Ipneg':
            
            return 0
        
        elif compartment == 'Ipos':
            
            return self.prob_chain(['Ipos', 'IQ', 'I', 'Ippos', 'Ipos'], param_obj)
        
        elif compartment == 'IQ':
            
            return self.prob_chain(['IQ', 'I', 'Ippos', 'Ipos', 'IQ'], param_obj)
        
    def prob_advance(self, compartment, param_obj):
        
        if compartment == "E":
            
            return self.prob_step("E", "I", param_obj) * 1 / (1 - self.prob_return("E", param_obj))
        
        elif compartment == "Epneg":
            
            return self.prob_step("Epneg", "Ipneg", param_obj) * 1 / (1 - self.prob_return("Epneg", param_obj))
        
        elif compartment == 'I':
            
            return self.prob_step('I', 'R', param_obj) * 1 / (1 - self.prob_return('I', param_obj))
        
        elif compartment == 'Ippos':
            
            return self.prob_step('Ippos', 'Rppos', param_obj) * 1 / (1 - self.prob_return('Ippos', param_obj))
        
        elif compartment == 'Ipneg':
            
            return self.prob_step('Ipneg', 'Rpneg', param_obj) * 1 / (1 - self.prob_return('Ipneg', param_obj))
        
        elif compartment == 'Ipos':
            
            return self.prob_step('Ipos', 'Rpos', param_obj) * 1 / (1 - self.prob_return('Ipos', param_obj))
        
        elif compartment == 'IQ':
            
            return self.prob_step('IQ', 'RQ', param_obj) * 1 / (1 - self.prob_return('IQ', param_obj))
                
    def prob_infected_in(self, compartment, param_obj):
        
        if compartment == 'S':
            
            return param_obj.Omega / (param_obj.Omega + param_obj.r_p)
        
        elif compartment == 'Spneg':
            
            return param_obj.r_p / (param_obj.Omega + param_obj.r_p)
        
        else:
            
            return None
        
    def expected_sojourn_time(self, compartment, param_obj):
        
        if compartment == 'I':
            
            return 1 / (param_obj.Npool * param_obj.r_p + param_obj.gamma) * 1 / (1 - self.prob_return("I", param_obj))
        
        elif compartment == 'Ippos':
            
            return 1 / (param_obj.Npool * param_obj.Omega + param_obj.gamma) * 1 / (1 - self.prob_return("Ippos", param_obj))
        
        elif compartment == 'Ipneg':
            
            return 1 / (param_obj.Npool * param_obj.Omega + param_obj.gamma) * 1 / (1 - self.prob_return("Ipneg", param_obj))

    def prob_hit(self, compartment_from, compartment_to, param_obj):
        
        if compartment_from == "E":
            
            if compartment_to == 'I':
                
                return self.prob_advance("E", param_obj) + self.prob_step('E', 'Epneg', param_obj) * self.prob_advance("Epneg", param_obj) * self.prob_chain(["Ipneg", "I"], param_obj)
            
            elif compartment_to == 'Ippos':
                
                return self.prob_advance("E", param_obj) * self.prob_step('I', "Ippos", param_obj) + self.prob_step('E', "Epneg", param_obj) * self.prob_advance("Epneg", param_obj) * self.prob_chain(['Ipneg', 'I', 'Ippos'], param_obj) 
            
            elif compartment_to == 'Ipneg':
                
                return self.prob_step('E', 'Epneg', param_obj) * self.prob_advance('Epneg', param_obj)
            
            elif compartment_to == 'Ipos':
                
                return self.prob_advance('E', param_obj) * self.prob_chain(['I', 'Ippos', 'Ipos'], param_obj) + self.prob_step('E', 'Epneg', param_obj) * self.prob_advance('Epneg', param_obj) * self.prob_chain(['Ipneg', 'I', 'Ippos', 'Ipos'], param_obj)
            
            elif compartment_to == 'IQ':
                
                return self.prob_advance('E', param_obj) * self.prob_chain(['I', 'Ippos', 'Ipos', 'IQ'], param_obj) + self.prob_step('E', 'Epneg', param_obj) * self.prob_advance('Epneg', param_obj) * self.prob_chain(['Ipneg', 'I', 'Ippos', 'Ipos', 'IQ'], param_obj)
        
        elif compartment_from == 'Epneg':
            
            if compartment_to == 'I':
                
                return self.prob_advance("Epneg", param_obj) * self.prob_step('Ipneg', "I", param_obj) + self.prob_step('Epneg', "E", param_obj) * self.prob_advance("E", param_obj)
            
            elif compartment_to == 'Ippos':
                
                return self.prob_advance("Epneg", param_obj) * self.prob_chain(["Ipneg", "I", "Ippos"], param_obj) + self.prob_step('Epneg', 'E', param_obj) * self.prob_advance("E", param_obj) * self.prob_step('I', "Ippos", param_obj) 
            
            elif compartment_to == 'Ipneg':
                
                return self.prob_advance('Epneg', param_obj)
            
            elif compartment_to == 'Ipos':
                
                return self.prob_advance('Epneg', param_obj) * self.prob_chain(['Ipneg', 'I', 'Ippos', 'Ipos'], param_obj) + self.prob_step('Epneg', 'E', param_obj) * self.prob_chain(['I', 'Ippos', 'Ipos'], param_obj)
            
            elif compartment_to == 'IQ':
                
                return self.prob_advance('Epneg', param_obj) * self.prob_chain(['Ipneg', 'I', 'Ippos', 'Ipos', 'IQ'], param_obj) + self.prob_step('Epneg', 'E', param_obj) * self.prob_advance('E', param_obj) * self.prob_chain(['I', 'Ippos', 'Ipos', 'IQ'], param_obj)
        
    def generate_R0_p(self, param_obj):
        
        expected_infected_time_individual_in_E = self.prob_hit('E', 'I', param_obj) * self.expected_sojourn_time('I', param_obj) + self.prob_hit('E', 'Ippos', param_obj) * self.expected_sojourn_time('Ippos', param_obj) + self.prob_hit('E', 'Ipneg', param_obj) * self.expected_sojourn_time('Ipneg', param_obj)
        expected_infected_time_individual_in_Epneg = self.prob_hit('Epneg', 'I', param_obj) * self.expected_sojourn_time('I', param_obj) + self.prob_hit('Epneg', 'Ippos', param_obj) * self.expected_sojourn_time('Ippos', param_obj) + self.prob_hit('Epneg', 'Ipneg', param_obj) * self.expected_sojourn_time('Ipneg', param_obj)
        
        return param_obj.beta * (
            self.prob_infected_in('S', param_obj) * expected_infected_time_individual_in_E +
            self.prob_infected_in('Spneg', param_obj) * expected_infected_time_individual_in_Epneg
        )
                                                             
    def final_size(self):
        
        # Returns the expected final size of the traditional SEIR epidemic
        
        f = lambda x: 1 - x - np.exp(-self.R0 * x)
        
        return optimize.newton(f, 0.99)

    def plot_R0_vs_r_v_varying_Npool(self, N_pool_vals, other_params_obj):
        
        fig = go.Figure()
        
        r_v_vals = np.linspace(0.0, 0.2)
        
        for Npool in Npool_vals:
            
            param_objs = [Parameters_SEIR_pool_testing(other_params_obj.R0, Npool, r_v, other_params_obj.t_E, other_params_obj.t_I, other_params_obj.t_P, other_params_obj.t_S, other_params_obj.t_Q, other_params_obj.t_linspace) for r_v in r_v_vals]
            R0_vals = [self.generate_R0_p(p_obj) for p_obj in param_objs]
            
            fig.add_trace(go.Scatter(x = r_v_vals,
                                     y = R0_vals,
                                     name = f"Pool size = {Npool}"
                                    )
                         )
            
        fig.update_layout(title = "Effect of increased testing on the pool reproductive number for differing pool sizes",
                          xaxis_title = r"$r_v$",
                          yaxis_title = r"$\mathbb{R}_0$"
                         )
        #fig.write_image('R0_different_testing_regimes.png', scale = 4)
        
        fig.show()

    def plot_R0_vs_t_P_varying_Npool(self, t_P_vals, Npool_vals, other_params_obj):
        
        fig = go.Figure()     
        
        for Npool in Npool_vals:
            
            param_objs = [Parameters_SEIR_pool_testing(other_params_obj.R0, Npool, other_params_obj.r_v, other_params_obj.t_E, other_params_obj.t_I, t_P, other_params_obj.t_S, other_params_obj.t_Q, other_params_obj.t_linspace) for t_P in t_P_vals]
            R0_vals = [self.generate_R0_p(p_obj) for p_obj in param_objs]
            
            fig.add_trace(go.Scatter(x = t_P_vals,
                                     y = R0_vals,
                                     name = f"Pool size = {Npool}"
                                    )
                         )
            
        
        fig.update_layout(title = "Effect of pool turn-about time on the pool reproductive number",
                          xaxis_title = r"$t_P$",
                          yaxis_title = r"$\mathbb{R}_0^P$"
                         )
        
        fig.show()
     
    def plot_R0_vs_t_S_varying_Npool(self, t_S_vals, Npool_vals, other_params_obj):
        
        fig = go.Figure()
        
        for Npool in Npool_vals:
            
            param_objs = [Parameters_SEIR_pool_testing(other_params_obj.R0, Npool, other_params_obj.r_v, other_params_obj.t_E, other_params_obj.t_I, other_params_obj.t_P, t_S, other_params_obj.t_Q, other_params_obj.t_linspace) for t_S in t_S_vals]
            R0_vals = [self.generate_R0_p(p_obj) for p_obj in param_objs]
            
            fig.add_trace(go.Scatter(x = t_S_vals,
                                     y = R0_vals,
                                     name = f"Pool size = {Npool}"
                                    )
                         )
            
        
        fig.update_layout(title = "Effect of standard test turn-about time on the pool reproductive number",
                          xaxis_title = r"$t_S$",
                          yaxis_title = r"$\mathbb{R}_0^P$"
                         )
        
        fig.show()
        
    def plot_R0_vs_t_tot_varying_Npool(self, t_tot_vals, Npool_vals, other_params_obj):
        
        fig = go.Figure()
        
        latent_proportion = other_params_obj.t_E / (other_params_obj.t_E + other_params_obj.t_I)
        
        for Npool in Npool_vals:
            
            param_objs = [Parameters_SEIR_pool_testing(other_params_obj.R0, Npool, other_params_obj.r_v, latent_proportion * t_tot, (1-latent_proportion) * t_tot, other_params_obj.t_P, other_params_obj.t_S, other_params_obj.t_Q, other_params_obj.t_linspace) for t_tot in t_tot_vals]
            R0_vals = [self.generate_R0_p(p_obj) for p_obj in param_objs]
            
            fig.add_trace(go.Scatter(x = t_tot_vals,
                                     y = R0_vals,
                                     name = f"Pool size = {Npool}"
                                    )
                         )
            
        
        fig.update_layout(title = "Effect of increasing total infected time on the pool reproductive number",
                          xaxis_title = r"$t_{tot}$",
                          yaxis_title = r"$\mathbb{R}_0^P$"
                         )
        
        fig.show()
        
    def plot_R0_vs_latent_proportion_varying_Npool(self, Npool_vals, other_params_obj):
        
        fig = go.Figure()
        
        latent_proportion_vals = np.linspace(0.01, 0.99, 1000)
        t_tot = other_params_obj.t_E + other_params_obj.t_I
        
        for Npool in Npool_vals:
            
            param_objs = [Parameters_SEIR_pool_testing(other_params_obj.R0, Npool, other_params_obj.r_v, latent_proportion * t_tot, (1-latent_proportion) * t_tot, other_params_obj.t_P, other_params_obj.t_S, other_params_obj.t_Q, other_params_obj.t_linspace) for latent_proportion in latent_proportion_vals]
            R0_vals = [self.generate_R0_p(p_obj) for p_obj in param_objs]
            
            fig.add_trace(go.Scatter(x = latent_proportion_vals,
                                     y = R0_vals,
                                     name = f"Pool size = {Npool}"
                                    )
                         )
            
        
        fig.update_layout(title = "Effect of increasing the proportion of infected time spend latent on the pool reproductive number",
                          xaxis_title = "Latent proportion",
                          yaxis_title = r"$\mathbb{R}_0^P$"
                         )
        
        fig.show()
   
    def code_checks(self, param_obj):
        
        # There are checks to see that the probabilities I've coded above satisfy certain conditions that
        # should be true mathematically
        
        #1. A person can either be in S or Sneg when infected
        
        if abs(self.prob_infected_in('S', param_obj) + self.prob_infected_in('Spneg', param_obj) - 1) > 0.0000001:
            
            print ("The probability of being infected in S added to the probability of being infected in Sneg doesn't add to 1")
            return 0      
        
        #2. The probability of not returning to a compartment in the latent or infectious phase equals the probability
        # that you recovered before you returned      
        
        prob_dont_return_to_E_1 = 1 - self.prob_return('E', param_obj)
        prob_dont_return_to_E_2 = self.prob_step('E', 'I', param_obj) + self.prob_chain(['E', 'Epneg', 'Ipneg'], param_obj)

        prob_dont_return_to_Epneg_1 = 1 - self.prob_return('Epneg', param_obj) 
        prob_dont_return_to_Epneg_2 = self.prob_step('Epneg', 'Ipneg', param_obj) + self.prob_chain(['Epneg', 'E', 'I'], param_obj)
        
        prob_dont_return_to_I_1 =  1 - self.prob_return('I', param_obj) 
        prob_dont_return_to_I_2 = self.prob_step('I', 'R', param_obj) + self.prob_chain(['I', 'Ippos', 'Rppos'], param_obj) + self.prob_chain(['I', 'Ippos', 'Ipos', 'Rpos'], param_obj) + self.prob_chain(['I', 'Ippos', 'Ipos', 'IQ', 'RQ'], param_obj)
        
        prob_dont_return_to_Ippos_1 = 1 - self.prob_return('Ippos', param_obj) 
        prob_dont_return_to_Ippos_2 = self.prob_step('Ippos', 'Rppos', param_obj) + self.prob_chain(['Ippos', 'Ipos', 'Rpos'], param_obj) + self.prob_chain(['Ippos', 'Ipos', 'IQ', 'RQ'], param_obj) + self.prob_chain(['Ippos', 'Ipos', 'IQ', 'I', 'R'], param_obj)
        
        prob_dont_return_to_Ipneg_1 = 1 - self.prob_return('Ipneg', param_obj)
        prob_dont_return_to_Ipneg_2 = self.prob_step('Ipneg', 'Rpneg', param_obj) + self.prob_step('Ipneg', 'I', param_obj) 
        
        prob_dont_return_to_Ipos_1 = 1 - self.prob_return('Ipos', param_obj)
        prob_dont_return_to_Ipos_2 = self.prob_step('Ipos', 'Rpos', param_obj) + self.prob_chain(['Ipos', 'IQ', 'RQ'], param_obj) + self.prob_chain(['Ipos', 'IQ', 'I', 'R'], param_obj) + self.prob_chain(['Ipos', 'IQ', 'I', 'Ippos', 'Rppos'], param_obj)
        
        prob_dont_return_to_IQ_1 = 1 - self.prob_return('IQ', param_obj) 
        prob_dont_return_to_IQ_2 = self.prob_step('IQ', 'RQ', param_obj) + self.prob_chain(['IQ', 'I', 'R'], param_obj) + self.prob_chain(['IQ', 'I', 'Ippos', 'Rppos'], param_obj) + self.prob_chain(['IQ', 'I', 'Ippos', 'Ipos', 'Rpos'], param_obj)
  
        if abs(prob_dont_return_to_E_1 - prob_dont_return_to_E_2) > 0.000001:
            
            print('The probability of not returning to E isn\'t correct')
            print(prob_dont_return_to_E_1)
            print(prob_dont_return_to_E_2)
        
        if abs(prob_dont_return_to_Epneg_1 - prob_dont_return_to_Epneg_2) > 0.00001:
            
            print('The probability of not returning to Epneg isn\'t correct')   
            print(prob_dont_return_to_Epneg_1)
            print(prob_dont_return_to_Epneg_2)
            
        if prob_dont_return_to_I_1 != prob_dont_return_to_I_2:
            
            print('The probability of not returning to I isn\'t correct')
            print(str(prob_dont_return_to_I_1))
            print(str(prob_dont_return_to_I_2))
        
        if prob_dont_return_to_Ippos_1 != prob_dont_return_to_Ippos_2:
            
            print('The probability of not returning to Ippos isn\'t correct') 
            
        if prob_dont_return_to_Ipneg_1 != prob_dont_return_to_Ipneg_2:
            
            print('The probability of not returning to Ipneg isn\'t correct')  
            
        if prob_dont_return_to_Ipos_1 != prob_dont_return_to_Ipos_2:
            
            print('The probability of not returning to Ipos isn\'t correct') 
            print(str(prob_dont_return_to_Ipos_1))
            print(str(prob_dont_return_to_Ipos_2))
        
        if prob_dont_return_to_IQ_1 != prob_dont_return_to_IQ_2:
            
            print('The probability of not returning to IQ isn\'t correct') 
            print(str(prob_dont_return_to_IQ_1))
            print(str(prob_dont_return_to_IQ_2))
            
        #3. Check the probabilities of return all equal
        
        if self.prob_return('E', param_obj) == self.prob_return('Epneg', param_obj) == False:
            
            print('Probabilities of return don\'t equal for latent compartments')
            print(str(self.prob_return('E', param_obj)))
            print(str(self.prob_return('Epneg', param_obj)))
            
        if self.prob_return('I', param_obj) == self.prob_return('Ippos', param_obj) == self.prob_return('Ipos', param_obj) == self.prob_return('IQ', param_obj) == False:
            
            print('Probabilities of return don\'t equal for infectious compartments')    
            
        #4. Check all the step probabilities add to one
        
        compartment_from_list = ['E', 'Epneg', 'I', 'Ippos', 'Ipos', 'IQ']
        compartment_to_list = ['E', 'Epneg', 'I', 'Ippos', 'Ipneg', 'Ipos', 'IQ', 'R', 'Rppos', 'Rpos', 'RQ']  
        
        for compartment_from in compartment_from_list:
            
            summed_probability = 0
            
            for compartment_to in compartment_to_list:
                
                summed_probability += self.prob_step(compartment_from, compartment_to, param_obj)
                
            if summed_probability != 1:
                
                print(compartment_from + ' has an issue with it\'s step probabilty')
                
        #5. Check that the advancement probabilities add to 1
        
        prob_advance_all_paths_E = self.prob_advance('E', param_obj) + self.prob_step('E', 'Epneg', param_obj) * self.prob_advance('Epneg', param_obj)
        prob_advance_all_paths_Epneg = self.prob_advance('Epneg', param_obj) + self.prob_step('Epneg', 'E', param_obj) * self.prob_advance('E', param_obj)
        
        prob_advance_all_paths_I = self.prob_advance('I', param_obj) + self.prob_step('I', 'Ippos', param_obj) * self.prob_advance('Ippos', param_obj) + self.prob_chain(['I', 'Ippos', 'Ipos'], param_obj) * self.prob_advance('Ipos', param_obj) + self.prob_chain(['I', 'Ippos', 'Ipos', 'IQ'], param_obj) * self.prob_advance('IQ', param_obj)
        prob_advance_all_paths_Ippos = self.prob_advance('Ippos', param_obj) + self.prob_step('Ippos', 'Ipos', param_obj) * self.prob_advance('Ipos', param_obj) + self.prob_chain(['Ippos', 'Ipos', 'IQ'], param_obj) * self.prob_advance('IQ', param_obj) + self.prob_chain(['Ippos', 'Ipos', 'IQ', 'I'], param_obj) * self.prob_advance('I', param_obj)
        prob_advance_all_paths_Ipneg = self.prob_advance('Ipneg', param_obj) + self.prob_step('Ipneg', 'I', param_obj) * self.prob_advance('I', param_obj) + self.prob_chain(['Ipneg', 'I', 'Ippos'], param_obj) * self.prob_advance('Ippos', param_obj) + self.prob_chain(['Ipneg', 'I', 'Ippos', 'Ipos'], param_obj) * self.prob_advance('Ipos', param_obj) + self.prob_chain(['Ipneg', 'I', 'Ippos', 'Ipos', 'IQ'], param_obj) * self.prob_advance('IQ', param_obj)
        prob_advance_all_paths_Ipos = self.prob_advance('Ipos', param_obj) + self.prob_step('Ipos', 'IQ', param_obj) * self.prob_advance('IQ', param_obj) + self.prob_chain(['Ipos', 'IQ', 'I'], param_obj) * self.prob_advance('I', param_obj) + self.prob_chain(['Ipos', 'IQ', 'I', 'Ippos'], param_obj) * self.prob_advance('Ippos', param_obj)
        prob_advance_all_paths_IQ = self.prob_advance('IQ', param_obj) + self.prob_step('IQ', 'I', param_obj) * self.prob_advance('I', param_obj) + self.prob_chain(['IQ', 'I', 'Ippos'], param_obj) * self.prob_advance('Ippos', param_obj) + self.prob_chain(['IQ', 'I', 'Ippos', 'Ipos'], param_obj) * self.prob_advance('Ipos', param_obj)
       
        if prob_advance_all_paths_E != 1:
            
            print('The advancement probabilities from the point of view of an individual in E don\'t sum to 1')
            print(str(prob_advance_all_paths_E))
            
        if prob_advance_all_paths_Epneg != 1:
            
            print('The advancement probabilities from the point of view of an individual in Epneg don\'t sum to 1')
            print(str(prob_advance_all_paths_Epneg))     

            
        if prob_advance_all_paths_I != 1:
            
            print('The advancement probabilities from the point of view of an individual in I don\'t sum to 1')
            print(str(prob_advance_all_paths_I))
            
        if prob_advance_all_paths_Ippos != 1:
            
            print('The advancement probabilities from the point of view of an individual in Ippos don\'t sum to 1')
            print(str(prob_advance_all_paths_Ippos))  
            
        if prob_advance_all_paths_Ipneg != 1:
            
            print('The advancement probabilities from the point of view of an individual in Ipneg don\'t sum to 1')
            print(str(prob_advance_all_paths_Ipneg))            
            
        if prob_advance_all_paths_Ipos != 1:
            
            print('The advancement probabilities from the point of view of an individual in Ipos don\'t sum to 1')
            print(str(prob_advance_all_paths_Ipos))     
            
        if prob_advance_all_paths_IQ != 1:
            
            print('The advancement probabilities from the point of view of an individual in IQ don\'t sum to 1')
            print(str(prob_advance_all_paths_IQ))    
            
            
        #6: Check that given some random distribution of the population among the different compartments
        #   that vector field satisfies the conservation of population principle, that is sums to one.
        
        for i in range(20):
            
            y = np.random.rand(20)
            y /= y.sum()
            
            sum_of_derivatives = sum(self.fun(0, y))
        
            if abs(sum_of_derivatives) > 0.000000001:
                
                print(f"Sum of vector field componants equals {sum_of_derivatives} instead of zero")

        
    

class Parameters_SEIR_standard_testing:
    
    def __init__(self, R0, r_v, t_E, t_I, t_S, t_Q, t_linspace):
        
        self.R0 = R0
        self.r_v = r_v
        self.sigma = 1 / t_E
        self.gamma = 1 / t_I
        self.omega = 1 / t_S
        self.delta = 1 / t_Q
        self.beta = R0 * self.gamma
        self.t_linspace = t_linspace

        if r_v * t_S >= 1:
            
            print("r_v * t_S must be less than 1")
            
        else:
        
            self.r_i = r_v / (1 - r_v * t_S)
        
class Parameters_SEIR_standard_testing:
    
    def __init__(self, R0, r_v, t_E, t_I, t_S, t_Q, t_linspace):
        
        self.R0 = R0
        self.r_v = r_v
        self.sigma = 1 / t_E
        self.gamma = 1 / t_I
        self.omega = 1 / t_S
        self.delta = 1 / t_Q
        self.beta = R0 * self.gamma
        self.t_linspace = t_linspace

        if r_v * t_S >= 1:
            
            print("r_v * t_S must be less than 1")
            
        else:
        
            self.r_i = r_v / (1 - r_v * t_S)
        
class SEIR_standard_testing:
    
    def __init__(self, param_obj):
        
        self.beta = param_obj.beta
        self.r_v = param_obj.r_v
        self.r_i = param_obj.r_i
        self.sigma = param_obj.sigma
        self.gamma = param_obj.gamma
        self.omega = param_obj.omega
        self.delta = param_obj.delta
        self.t_linspace = param_obj.t_linspace
        self.R0 = param_obj.R0
        
    def fun(self, t, y):
        
        S, Sneg = y[0:2]
        E, Epos, Eneg, EQ = y[2:6]
        I, Ipos, Ineg, IQ = y[6:10]
        R, Rpos, Rneg, RQ = y[10:14]
        
        
        d_S_dt = self.omega * Sneg  - self.beta * S * (I + Ipos + Ineg) - self.r_i * S
        d_Sneg_dt = self.r_i * S  - self.beta * Sneg * (I + Ipos + Ineg) - self.omega * Sneg
        d_E_dt = self.beta * S * (I + Ipos + Ineg) + self.omega * Eneg + self.delta * EQ - self.r_i * E - self.sigma * E
        d_Epos_dt = self.r_i * E - self.omega * Epos - self.sigma * Epos
        d_Eneg_dt = self.beta * Sneg * (I + Ipos + Ineg) - (self.omega + self.sigma) * Eneg
        d_EQ_dt = self.omega * Epos - self.delta * EQ - self.sigma * EQ
        d_I_dt = self.sigma * E + self.omega * Ineg + self.delta * IQ - self.r_i * I - self.gamma * I
        d_Ipos_dt = self.sigma * Epos + self.r_i * I - self.omega * Ipos - self.gamma * Ipos
        d_Ineg_dt = self.sigma * Eneg - (self.omega + self.gamma) * Ineg
        d_IQ_dt = self.sigma * EQ + self.omega * Ipos - self.delta * IQ - self.gamma * IQ
        d_R_dt = self.gamma * I  + self.omega * Rneg + self.delta * RQ - self.r_i * R
        d_Rpos_dt = self.gamma * Ipos - self.omega * Rpos
        d_Rneg_dt = self.gamma * Ineg + self.r_i * R  - self.omega * Rneg
        d_RQ_dt = self.gamma * IQ + self.omega * Rpos - self.delta * RQ
        
        dydt = [d_S_dt, d_Sneg_dt, d_E_dt, d_Epos_dt, d_Eneg_dt, d_EQ_dt, d_I_dt, d_Ipos_dt, d_Ineg_dt, d_IQ_dt, d_R_dt, d_Rpos_dt, d_Rneg_dt, d_RQ_dt]
        
        return dydt
    
    def generate_solution(self, y0 = None, method = 'LSODA'):
        
        # This generates a solution over a linear time space t_linspace
        
        if y0 == None:
            y0 = [7999999/8000000, 0, 1/8000000, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        t0, tf = self.t_linspace[0], self.t_linspace[-1]
        return solve_ivp(self.fun, (t0, tf), y0, method, t_eval = self.t_linspace)
    
    def S(self, compartment_string = None):
        
        if compartment_string == None:
            
            return self.sol[0]
        
        
        elif compartment_string == 'neg':
            
            return self.sol[1]
    
    def E(self, compartment_string = None):
        
        if compartment_string == None:
            
            return self.sol[2]
        
        elif compartment_string == 'pos':
            
            return self.sol[3]
        
        elif compartment_string == 'neg':
            
            return self.sol[4]        
        
        elif compartment_string == 'Q':
            
            return self.sol[5] 
        
    def I(self, compartment_string = None):
        
        if compartment_string == None:
            
            return self.sol[6]
        
        elif compartment_string == 'pos':
            
            return self.sol[7]
        
        elif compartment_string == 'neg':
            
            return self.sol[8]    
        
        elif compartment_string == 'Q':
            
            return self.sol[9]
        
    def R(self, compartment_string = None):
        
        if compartment_string == None:
            
            return self.sol[10]
        
        elif compartment_string == 'pos':
            
            return self.sol[11]
        
        elif compartment_string == 'neg':
            
            return self.sol[12]
        
        elif compartment_string == 'Q':
            
            return self.sol[13]
            
    def r_SEIR_traditional(self):
        
        return (1/2) * (- (self.sigma + self.gamma) + np.sqrt((self.sigma + self.gamma)**2 + 4 * self.sigma * (self.beta - self.gamma)))  
    
    def E_proportion_infected_population_limiting_SEIR_traditional(self):
        
        r = self.r_SEIR_traditional()
        
        return (self.gamma + r) / (self.sigma + self.gamma + r)

    def I_proportion_infected_population_limiting_SEIR_traditional(self):
        
        r = self.r_SEIR_traditional()
        
        return (self.sigma) / (self.sigma + self.gamma + r)      
    
    def prob_step(self, compartment_from, compartment_to, param_obj):
        
        if compartment_from == 'E':
            
            if compartment_to == 'Epos':
                
                return param_obj.r_i / (param_obj.r_i + param_obj.sigma)
            
            elif compartment_to == 'I':
                
                return param_obj.sigma / (param_obj.r_i + param_obj.sigma)
            
            else:
                
                return 0
            
        elif compartment_from == 'Epos':
            
            if compartment_to == 'EQ':
                
                return param_obj.omega / (param_obj.omega + param_obj.sigma)
            
            elif compartment_to == 'Ipos':
                
                return param_obj.sigma / (param_obj.omega + param_obj.sigma)
            
            else:
                
                return 0
            
        elif compartment_from == 'Eneg':
            
            if compartment_to == 'E':
                
                return param_obj.omega / (param_obj.omega + param_obj.sigma)
            
            elif compartment_to == 'Ineg':
                
                return param_obj.sigma / (param_obj.omega + param_obj.sigma)
            
            else:
                
                return 0
            
        elif compartment_from == 'EQ':
            
            if compartment_to == 'E':
                
                return param_obj.delta / (param_obj.delta + param_obj.sigma)
            
            elif compartment_to == 'IQ':
                
                return param_obj.sigma / (param_obj.delta + param_obj.sigma)
            
            else:
                
                return 0
            
        elif compartment_from == 'I':
            
            if compartment_to == 'Ipos':
                
                return param_obj.r_i / (param_obj.r_i + param_obj.gamma)
            
            elif compartment_to == 'R':
                
                return param_obj.gamma / (param_obj.r_i + param_obj.gamma)
            
            else:
                
                return 0
            
        elif compartment_from == 'Ipos':
            
            if compartment_to == 'IQ':
                
                return param_obj.omega / (param_obj.omega + param_obj.gamma)
            
            elif compartment_to == 'Rpos':
                
                return param_obj.gamma / (param_obj.omega + param_obj.gamma)
            
            else:
                
                return 0
            
        elif compartment_from == 'Ineg':
            
            if compartment_to == 'I':
                
                return param_obj.omega / (param_obj.omega + param_obj.gamma)
            
            elif compartment_to == 'Rneg':
                
                return param_obj.gamma / (param_obj.omega + param_obj.gamma)
            
            else:
                
                return 0
        
        elif compartment_from == 'IQ':
            
            if compartment_to == 'I':
                
                return param_obj.delta / (param_obj.delta + param_obj.gamma)
            
            elif compartment_to == 'RQ':
                
                return param_obj.gamma / (param_obj.delta + param_obj.gamma)
            
            else:
                
                return 0
            
    def prob_chain(self, compartments, param_obj):
        
        chain_probability = 1
        
        for i in range(len(compartments) - 1):
            
            compartment_from = compartments[i]
            compartment_to = compartments[i + 1]
            
            chain_probability = chain_probability * self.prob_step(compartment_from, compartment_to, param_obj)
            
        return chain_probability
          
    def prob_return(self, compartment, param_obj):
        
        if compartment == "E":
            
            return self.prob_chain(['E', 'Epos', 'EQ', 'E'], param_obj)
        
        elif compartment == 'Epos':
            
            return self.prob_chain(['Epos', 'EQ', 'E', 'Epos'], param_obj)
        
        elif compartment == 'Eneg':
            
            return 0
        
        elif compartment == 'EQ':
            
            return self.prob_chain(['EQ', 'E', 'Epos', 'EQ'], param_obj)
        
        elif compartment == 'I':
            
            return self.prob_chain(['I', 'Ipos', 'IQ', 'I'], param_obj)
        
        elif compartment == 'Ipos':
            
            return self.prob_chain(['Ipos', 'IQ', 'I', 'Ipos'], param_obj)
        
        elif compartment == 'Ineg':
            
            return 0
        
        elif compartment == 'IQ':
            
            return self.prob_chain(['IQ', 'I', 'Ipos', 'IQ'], param_obj)    
        
    def prob_advance(self, compartment, param_obj):
        
        if compartment == "E":
            
            return self.prob_step('E', 'I', param_obj) * 1 / (1 - self.prob_return("E", param_obj))
        
        elif compartment == "Epos":
            
            return self.prob_step('Epos', 'Ipos', param_obj) * 1 / (1 - self.prob_return("Epos", param_obj))
        
        elif compartment == "Eneg":
            
            return self.prob_step('Eneg', 'Ineg', param_obj) * 1 / (1 - self.prob_return("Eneg", param_obj))
        
        elif compartment == "EQ":
            
            return self.prob_step('EQ', 'IQ', param_obj) * 1 / (1 - self.prob_return("EQ", param_obj))
        
        elif compartment == "I":
            
            return self.prob_step('I', 'R', param_obj) * 1 / (1 - self.prob_return("I", param_obj))
        
        elif compartment == "Ipos":
            
            return self.prob_step('Ipos', 'Rpos', param_obj) * 1 / (1 - self.prob_return("Ipos", param_obj))
        
        elif compartment == "Ineg":
            
            return self.prob_step('Ineg', 'Rneg', param_obj) * 1 / (1 - self.prob_return("Ineg", param_obj))
        
        elif compartment == "IQ":
            
            return self.prob_step('IQ', 'RQ', param_obj) * 1 / (1 - self.prob_return("IQ", param_obj))
        
    def expected_sojourn_time(self, compartment, param_obj):
        
        if compartment == 'I':
            
            return 1 / (param_obj.r_i + param_obj.gamma) * 1 / (1 - self.prob_return("I", param_obj))
        
        elif compartment == 'Ipos':
            
            return 1 / (param_obj.omega + param_obj.gamma) * 1 / (1 - self.prob_return("Ipos", param_obj))
        
        elif compartment == 'Ineg':
            
            return 1 / (param_obj.omega + param_obj.gamma) * 1 / (1 - self.prob_return("Ineg", param_obj))
        
    def prob_infected_in(self, compartment, param_obj):
        
        if compartment == 'S':
            
            return param_obj.omega / (param_obj.omega + param_obj.r_i)
        
        elif compartment == 'Sneg':
            
            return param_obj.r_i / (param_obj.omega + param_obj.r_i)
        
        else:
            
            return None
        
    def prob_hit(self, compartment_from, compartment_to, param_obj):
        
        if compartment_from == "E":
            
            if compartment_to == 'I':
                
                return self.prob_advance("E", param_obj) + self.prob_step('E', 'Epos', param_obj) * self.prob_advance("Epos", param_obj) * self.prob_chain(["Ipos", "IQ", "I"], param_obj) + self.prob_chain(["E", "Epos", "EQ"], param_obj) * self.prob_advance("EQ", param_obj) * self.prob_step('IQ', "I", param_obj)
            
            elif compartment_to == 'Ipos':
                
                return self.prob_advance("E", param_obj) * self.prob_step('I', "Ipos", param_obj) + self.prob_step('E', "Epos", param_obj) * self.prob_advance("Epos", param_obj) + self.prob_chain(["E", "Epos", "EQ"], param_obj) * self.prob_advance("EQ", param_obj) * self.prob_chain(["IQ", "I", "Ipos"], param_obj) 
        
        elif compartment_from == 'Eneg':
            
            if compartment_to == 'I':
                
                return self.prob_advance("Eneg", param_obj) * self.prob_step('Ineg', "I", param_obj) + self.prob_step('Eneg', "E", param_obj) * self.prob_advance("E", param_obj) + self.prob_chain(["Eneg", "E", "Epos"], param_obj)* self.prob_advance("Epos", param_obj) * self.prob_chain(["Ipos", "IQ", "I"], param_obj) + self.prob_chain(["Eneg", "E", "Epos", "EQ"], param_obj) * self.prob_advance("EQ", param_obj) * self.prob_step('IQ', 'I', param_obj)
            
            elif compartment_to == 'Ipos':
                
                return self.prob_advance("Eneg", param_obj) * self.prob_chain(["Ineg", "I", "Ipos"], param_obj) + self.prob_step('Eneg', 'E', param_obj) * self.prob_advance("E", param_obj) * self.prob_step('I', "Ipos", param_obj) + self.prob_chain(["Eneg", "E", "Epos"], param_obj) * self.prob_advance("Epos", param_obj) + self.prob_chain(["Eneg", "E", "Epos", "EQ"], param_obj) * self.prob_advance("EQ", param_obj) * self.prob_chain(["IQ", "I", "Ipos"], param_obj)
            
            elif compartment_to == 'Ineg':
                
                return self.prob_step('Eneg', 'Ineg', param_obj)
            
    def generate_R0_s(self, param_obj):
        
        # E_X_Y is expected time spent in X given an individual was infected in Y
        
        E_I_S = self.prob_infected_in('S', param_obj) * self.prob_hit('E', 'I', param_obj) * self.expected_sojourn_time('I', param_obj)
        E_Ipos_S = self.prob_infected_in('S', param_obj) * self.prob_hit('E', 'Ipos', param_obj) * self.expected_sojourn_time('Ipos', param_obj)
        E_I_Sneg = self.prob_infected_in('Sneg', param_obj) * self.prob_hit('Eneg', 'I', param_obj) * self.expected_sojourn_time('I',  param_obj)
        E_Ipos_Sneg = self.prob_infected_in('Sneg', param_obj) * self.prob_hit('Eneg', 'Ipos', param_obj) * self.expected_sojourn_time('Ipos', param_obj)
        E_Ineg_Sneg = self.prob_infected_in('Sneg', param_obj) * self.prob_hit('Eneg', 'Ineg', param_obj) * self.expected_sojourn_time('Ineg', param_obj)
        
        return param_obj.beta * (E_I_S + E_Ipos_S + E_I_Sneg + E_Ipos_Sneg + E_Ineg_Sneg)
    
    def code_checks(self, param_obj):
        
        # There are checks to see that the probabilities I've coded above satisfy certain conditions that
        # should be true mathematically
        
        #1. A person can either be in S or Sneg when infected
        
        if self.prob_infected_in('S', param_obj) + self.prob_infected_in('Sneg', param_obj) != 1:
            
            print ("The probability of being infected in S added to the probability of being infected in Sneg doesn't add to 1")
            return 0
        
        #2. The probability of not returning to a compartment in the latent or infectious phase equals the probability
        # that you recovered before you returned
        
        
        prob_dont_return_to_E_1 = 1 - self.prob_return('E', param_obj)
        prob_dont_return_to_E_2 = self.prob_step('E', 'I', param_obj) + self.prob_chain(['E', 'Epos', 'Ipos'], param_obj) + self.prob_chain(['E', 'Epos', 'EQ', 'IQ'], param_obj)

        prob_dont_return_to_Epos_1 = 1 - self.prob_return('Epos', param_obj) 
        prob_dont_return_to_Epos_2 = self.prob_step('Epos', 'Ipos', param_obj) + self.prob_chain(['Epos', 'EQ', 'IQ'], param_obj) + self.prob_chain(['Epos', 'EQ', 'E', 'I'], param_obj)
        
        prob_dont_return_to_EQ_1 = 1 - self.prob_return('EQ', param_obj) 
        prob_dont_return_to_EQ_2 = self.prob_step('EQ', 'IQ', param_obj) + self.prob_chain(['EQ', 'E', 'I'], param_obj) + self.prob_chain(['EQ', 'E', 'Epos', 'Ipos'], param_obj)
        
        prob_dont_return_to_I_1 =  1 - self.prob_return('I', param_obj) 
        prob_dont_return_to_I_2 = self.prob_step('I', 'R', param_obj) + self.prob_chain(['I', 'Ipos', 'Rpos'], param_obj) + self.prob_chain(['I', 'Ipos', 'IQ', 'RQ'], param_obj)
        
        prob_dont_return_to_Ipos_1 = 1 - self.prob_return('Ipos', param_obj) 
        prob_dont_return_to_Ipos_2 = self.prob_step('Ipos', 'Rpos', param_obj) + self.prob_chain(['Ipos', 'IQ', 'RQ'], param_obj) + self.prob_chain(['Ipos', 'IQ', 'I', 'R'], param_obj)
        
        prob_dont_return_to_IQ_1 = 1 - self.prob_return('IQ', param_obj) 
        prob_dont_return_to_IQ_2 = self.prob_step('IQ', 'RQ', param_obj) + self.prob_chain(['IQ', 'I', 'R'], param_obj) + self.prob_chain(['IQ', 'I', 'Ipos', 'Rpos'], param_obj)
  

        if prob_dont_return_to_E_1 != prob_dont_return_to_E_2:
            
            print('The probability of not returning to E isn\'t correct')
            print(prob_dont_return_to_E_1)
            print(prob_dont_return_to_E_2)
        
        if prob_dont_return_to_Epos_1 != prob_dont_return_to_Epos_2:
            
            print('The probability of not returning to Epos isn\'t correct')  
            print(prob_dont_return_to_Epos_1)
            print(prob_dont_return_to_Epos_2)
        
        if prob_dont_return_to_EQ_1 != prob_dont_return_to_EQ_2:
            
            print('The probability of not returning to EQ isn\'t correct')  
            print(str(1 - self.prob_return('EQ', param_obj)))
            print(str(self.prob_step('EQ', 'IQ', param_obj) + self.prob_chain(['EQ', 'E', 'I'], param_obj) + self.prob_chain(['EQ', 'E', 'Epos', 'Ipos'], param_obj)))
            
        if prob_dont_return_to_I_1 != prob_dont_return_to_I_2:
            
            print('The probability of not returning to I isn\'t correct')
            print(str(1 - self.prob_return('I', param_obj)))
            print(str(self.prob_step('I', 'R', param_obj) + self.prob_chain(['I', 'Ipos', 'Rpos'], param_obj) + self.prob_chain(['I', 'Ipos', 'IQ', 'RQ'], param_obj)))
        
        if prob_dont_return_to_Ipos_1 != prob_dont_return_to_Ipos_2:
            
            print('The probability of not returning to Ipos isn\'t correct') 
            print(str(1 - self.prob_return('Ipos', param_obj)))
            print(str( self.prob_step('Ipos', 'Rpos', param_obj) + self.prob_chain(['Ipos', 'IQ', 'RQ'], param_obj) + self.prob_chain(['Ipos', 'IQ', 'I', 'R'], param_obj)))
        
        if prob_dont_return_to_IQ_1 != prob_dont_return_to_IQ_2:
            
            print('The probability of not returning to IQ isn\'t correct') 
            print(str(1 - self.prob_return('IQ', param_obj)))
            print(str(self.prob_step('IQ', 'RQ', param_obj) + self.prob_chain(['IQ', 'I', 'R'], param_obj) + self.prob_chain(['IQ', 'I', 'Ipos', 'Rpos'], param_obj)))
            
        #3. Check the probabilities of return all equal
        
        if self.prob_return('E', param_obj) == self.prob_return('Epos', param_obj) == self.prob_return('EQ', param_obj) == False:
            
            print('Probabilities of return don\'t equal for latent compartments')
            
        if self.prob_return('I', param_obj) == self.prob_return('Ipos', param_obj) == self.prob_return('IQ', param_obj) == False:
            
            print('Probabilities of return don\'t equal for infectious compartments')    
            
        #4. Check all the step probabilities add to one
        
        compartment_from_list = ['E', 'Epos', 'Eneg', 'EQ', 'I', 'Ipos', 'Ineg', 'IQ']
        compartment_to_list = ['E', 'Epos', 'Eneg', 'EQ', 'I', 'Ipos', 'Ineg', 'IQ', 'R', 'Rpos', 'Rneg', 'RQ']  
        
        for compartment_from in compartment_from_list:
            
            summed_probability = 0
            
            for compartment_to in compartment_to_list:
                
                summed_probability += self.prob_step(compartment_from, compartment_to, param_obj)
                
            if summed_probability != 1:
                
                print(compartment_from + ' has an issue with it\'s step probabilty')
                
        #5. Check that the advancement probabilities add to 1
        
        prob_advance_all_paths_E = self.prob_advance('E', param_obj) + self.prob_step('E', 'Epos', param_obj) * self.prob_advance('Epos', param_obj) + self.prob_chain(['E', 'Epos', 'EQ'], param_obj) * self.prob_advance('EQ', param_obj)
        prob_advance_all_paths_Epos = self.prob_advance('Epos', param_obj) + self.prob_step('Epos', 'EQ', param_obj) * self.prob_advance('EQ', param_obj) + self.prob_chain(['Epos', 'EQ', 'E'], param_obj) * self.prob_advance('E', param_obj)
        prob_advance_all_paths_Eneg = self.prob_advance('Eneg', param_obj) + self.prob_step('Eneg', 'E', param_obj) * self.prob_advance('E', param_obj) + self.prob_chain(['Eneg', 'E', 'Epos'], param_obj) * self.prob_advance('Epos', param_obj) + self.prob_chain(['Eneg', 'E', 'Epos', 'EQ'], param_obj) * self.prob_advance('EQ', param_obj)
        prob_advance_all_paths_EQ = self.prob_advance('EQ', param_obj) + self.prob_step('EQ', 'E', param_obj) * self.prob_advance('E', param_obj) + self.prob_chain(['EQ', 'E', 'Epos'], param_obj) * self.prob_advance('Epos', param_obj)
        
        prob_advance_all_paths_I = self.prob_advance('I', param_obj) + self.prob_step('I', 'Ipos', param_obj) * self.prob_advance('Ipos', param_obj) + self.prob_chain(['I', 'Ipos', 'IQ'], param_obj) * self.prob_advance('IQ', param_obj)
        prob_advance_all_paths_Ipos = self.prob_advance('Ipos', param_obj) + self.prob_step('Ipos', 'IQ', param_obj) * self.prob_advance('IQ', param_obj) + self.prob_chain(['Ipos', 'IQ', 'I'], param_obj) * self.prob_advance('I', param_obj)
        prob_advance_all_paths_Ineg = self.prob_advance('Ineg', param_obj) + self.prob_step('Ineg', 'I', param_obj) * self.prob_advance('I', param_obj) + self.prob_chain(['Ineg', 'I', 'Ipos'], param_obj) * self.prob_advance('Ipos', param_obj) + self.prob_chain(['Ineg', 'I', 'Ipos', 'IQ'], param_obj) * self.prob_advance('IQ', param_obj)
        prob_advance_all_paths_IQ = self.prob_advance('IQ', param_obj) + self.prob_step('IQ', 'I', param_obj) * self.prob_advance('I', param_obj) + self.prob_chain(['IQ', 'I', 'Ipos'], param_obj) * self.prob_advance('Ipos', param_obj)
       
        if prob_advance_all_paths_E != 1:
            
            print('The advancement probabilities from the point of view of an individual in E don\'t sum to 1')
            print(str(prob_advance_all_paths_E))
            
        if prob_advance_all_paths_Epos != 1:
            
            print('The advancement probabilities from the point of view of an individual in Epos don\'t sum to 1')
            print(str(prob_advance_all_paths_Epos))     
            
        if prob_advance_all_paths_Eneg != 1:
            
            print('The advancement probabilities from the point of view of an individual in Eneg don\'t sum to 1')
            print(str(prob_advance_all_paths_Eneg))
            
        if prob_advance_all_paths_EQ != 1:
            
            print('The advancement probabilities from the point of view of an individual in EQ don\'t sum to 1')
            print(str(prob_advance_all_paths_EQ))  
            
        if prob_advance_all_paths_I != 1:
            
            print('The advancement probabilities from the point of view of an individual in I don\'t sum to 1')
            print(str(prob_advance_all_paths_I))
            
        if prob_advance_all_paths_Ipos != 1:
            
            print('The advancement probabilities from the point of view of an individual in Ipos don\'t sum to 1')
            print(str(prob_advance_all_paths_Ipos))     
            
        if prob_advance_all_paths_Ineg != 1:
            
            print('The advancement probabilities from the point of view of an individual in Ineg don\'t sum to 1')
            print(str(prob_advance_all_paths_Ineg))
            
        if prob_advance_all_paths_IQ != 1:
            
            print('The advancement probabilities from the point of view of an individual in IQ don\'t sum to 1')
            print(str(prob_advance_all_paths_IQ))               
    
    def final_size(self):
        
        f = lambda x: 1 - x - np.exp(-self.R0 * x)
        
        return optimize.newton(f, 0.99)
    
    def final_size_numerical(self):
        
        # Returns the final size of the epidemic as calculated using the numerical solution. 
        # Made as a check to see the analytic form in final_size is correct
        
        return 1 - (self.S()[-1] + self.S("neg")[-1])
    
    def parameter_that_returns_final_size(self, parameter_string, final_size_proportion, visual_check = False):
        
        # Returns the parameter value needed such that you get that final_size_proportion
        
        if parameter_string == "r_i":
            
            # Means we are looking for parameter values of r_i that will return a 
            # final size proportion equal to final_size_proportion
            
            f = lambda r_i: 1 - final_size_proportion - np.exp(-self.generate_R0(self.beta, r_i, self.omega, self.delta, self.sigma, self.gamma) * final_size_proportion)

            optimal_r_i = optimize.newton(f, 0.1)
            
            if visual_check == True:

                r_i_vals = np.linspace(0.01, 1, 1000)
                plt.plot(r_i_vals, [f(r_i) for r_i in r_i_vals])
                plt.scatter(optimal_r_i, f(optimal_r_i))
                plt.show()
                                                               
            return optimal_r_i
        
        elif parameter_string == "t_S":
            
            # Looking for values of the turnaround time for tests t_S that will return a final size proportion
            # equal to final_size_proportion
            
            f = lambda t_S: 1 - final_size_proportion - np.exp(-self.generate_R0(self.beta, self.r_i, 1 / t_S, self.delta, self.sigma, self.gamma) * final_size_proportion)
            
            
            try:
                optimal_t_S = optimize.newton(f, 0.1)
                
            except Exception as e:
                
                # Handle a runtime error here like failure to converge
                
                print(f"An error occurred: {e}")
                t_S_vals = np.linspace(0.01, 20, 1000)
                plt.plot(t_S_vals, f(t_S_vals))
                plt.show()
            
            except:
                
                if visual_check == True:
                
                    t_S_vals = np.linspace(0.01, optimal_t_S + 5, 1000)
                    plt.plot(t_S_vals, f(t_S_vals))
                    plt.scatter(optimal_t_S, f(optimal_t_S))
                    plt.show()
                    
                    return optimal_t_S
                
                else:   
                
                    return optimal_t_S
                