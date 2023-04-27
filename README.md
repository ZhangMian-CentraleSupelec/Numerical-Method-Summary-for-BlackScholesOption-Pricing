# Numerical-Method-Summary-for-BlackScholesOption-Pricing


This notebook use most existed popular numerical methods and compare the accuracy of the method. And also, it practice the decorator with multi-thread techniques to boost the performance.

###
- Black-Scholes closed formula
    - Conditional Expectation
    - Forrier Transform
- Monte Carlo
    - Euler Scheme
    - Milstein Scheme
    - Longstaff-Schwarz
    - Variance Reduction
        - Antithetique Variant
        - Variable Control
        - Importance sampling (For out-of-money option principally)
        - Quasi Monte Carlo 
- Tree model
    - American option pricing
- Finite difference
    - Explicit Euler
    - Implicit Euler
    - Crank Nicolson

### Black-Scholes Conditional Expectation Formula

Black-Scholes formula:

$$P(t,S_t) = S(t,S_t)\omega N(\omega d_1) + K \omega N(\omega d_2)$$ 

$\omega$ equals to 1 for call, -1 for put.

$$d_1 = \frac{log(\frac{S_t}{K}) + (r+\frac{\sigma ^2}{2})(T-t)}{\sigma \sqrt{T-t}}$$

$$d_2 = \frac{log(\frac{S_t}{K}) + (r-\frac{\sigma ^2}{2})(T-t)}{\sigma \sqrt{T-t}}$$

### Black-Scholes Characteristic Function Formula
$$C(t,S_t) = D(t,T)E(h(S_T))$$

$$= D(t,T)\int^{\infty}_{K}(S_T-K)f(S_T)dS_T$$

$$= D(t,T)\int^{\infty}_{log(K)}(e^X-K)f(X)dX$$

Based on the Gil-Pelaez theorem:

$$F(x) = \frac{1}{2} - \frac{1}{\pi}\int^{\infty}_{0}Re(\frac{e^{-iwx}\varphi_X(w)}{iw})dw$$

here we have:
$$C_0 = S_0\Pi_1 - D(t,T)K\Pi_2$$

$$\varphi_{lnS}(w) = exp({iw(ln(S_0)+(r-\frac{1}{2}\sigma^2)t) - \frac{1}{2}w^2\sigma^2t})$$

$$\Pi_1 = \frac{1}{2} - \frac{1}{\pi}\int^{\infty}_{0}Re(\frac{e^{-iwln(K)}\varphi_{lnS}(w-i)}{iw\varphi_{lnS}(-i)})dw$$

$$ \Pi_2 = \frac{1}{2} - \frac{1}{\pi}\int^{\infty}_{0}Re(\frac{e^{-iwln(K)}\varphi_{lnS}(w)}{iw})dw $$

### Greeks

Assume no dividend.

$$\Delta = \frac{\partial V}{\partial S} = \omega N(\omega d_1)$$

$$\Gamma = \frac{\partial^2 V}{\partial S^2} = \frac{N(d_1)}{S\sigma\sqrt{T-t}}$$

$$\theta = \frac{\partial V}{\partial t} = -\frac{SN(d_1)\sigma}{2\sqrt{T-t}}-\omega rKe^{r(T-t)}N(\omega d_2)$$

$$\vartheta  = \frac{\partial V}{\partial \sigma} = SN(d_1)\sqrt{T-t}$$

$$\rho = \frac{\partial V}{\partial r} =  \omega K(T-t)e^{-r(T-t)}N(\omega d_2)$$


![image](https://user-images.githubusercontent.com/110284601/234842202-8e2b6efb-094a-48b0-94e7-978b25bcd391.png)

### Monte Carlo

- Boost Performance for Monte Carlo
    - Joblib / Multiprocess for Boosting
    - Decorator of Monte Carlo
- Discritization
    - Euler-Maruyama Scheme
    - Milstein Scheme
- American Case: Longstaff-Schwartz Algorithm
- Variance Reduction
    - Anti Variant
    - Variable control
    - Importance Sampling
    - Quasi Monte Carlo

### Boost the performance of Monte Carlo

In this notebook we basically use 2 methods to boost the speed:

- joblib to realize parallel computing.
- numpy vectorize to use vectorize calculation.

Also there are many other methods, like Numba and Cython can be used to compile Python code into lower-level languages (such as C and C++) to enhance execution efficiency. But it's not in the content of this notebook.

### Decorator for Monte Carlo

Here we create an Decorater, which can analyse the vairance and confidence interval with the increase of interations.

### Euler Scheme for Black-Scholes Model

- Normal case: $$S_t = S_0 + rS_t \Delta t + \sigma S_0\sqrt{\Delta t}Z$$ 

- More precise: $$log(S_{t})-log(S_{0}) = (r- \frac{\sigma^{2}}{2})t + \sigma \sqrt{t}z$$

- same as $$S_{t} = S_{0}exp((r-\frac{1}{2}\sigma ^2)t+\sigma \sqrt{t} Z)$$

- Normal Case:

![image](https://user-images.githubusercontent.com/110284601/234846018-596f13a6-d149-466e-a60c-a9d4991f66d5.png)

- LogNormal Case:

![image](https://user-images.githubusercontent.com/110284601/234842451-6097e7f1-9d86-41f6-b4e1-92fc4db23800.png)

### Milstein Scheme Introduction

In the normal case: 
$$ dS_t = \mu(S_t)dt + \sigma(S_t)dW_t$$  
$$ S_{t+dt} = S_t + \int_{t}^{t+dt}\mu_s S_t ds + \int_{t}^{t+dt}\sigma_s S_tdWs$$  

In Milstein Scheme, we use ito's lemma to $\sigma(t)$ and $\mu(t)$.

$$d\mu(S_t) = \mu_t'dS_t + \frac{1}{2}\mu_t''(dS_t)^2$$ 
$$= (\mu_t'\mu_t + \frac{1}{2}\mu_t''\sigma_t^2)dt + (\mu_t'\sigma_t)dW_t$$

$$d\sigma(S_t) = \sigma_t'dS_t + \frac{1}{2}\sigma_t''(dS_t)^2$$ 
$$= (\sigma_t'\mu_t + \frac{1}{2}\sigma_t''\sigma_t^2)dt + (\sigma_t'\sigma_t)dW_t$$

$$\mu_s = \mu_t + \int_{t}^{s}(\mu'_u\mu_u+\frac{1}{2}\mu_u''\sigma_u^2)du + \int_{t}^{s}(\mu_u'\sigma_u)dW_u$$ 
$$\sigma_s = \sigma_t + \int_{t}^{s}(\sigma_u'\mu_u + \frac{1}{2}\sigma_u''\sigma_u^2)du + \int_{t}^{s}(\sigma_u'\sigma_u)dW_u$$

$$S_{t+dt} = S_t + \int_{t}^{t+dt} (\mu_t + \int_{t}^{s} ((\mu_u'\mu_u + \frac{1}{2}\mu_u''\sigma_u^2)du + (\mu_u'\sigma_u)dW_u))ds)+ \int_{t}^{t+dt}(\sigma_t+\int_{t}^{s}(\sigma_u'\mu_u + \frac{1}{2}\sigma_u''\sigma_u^2)du + (\sigma_u'\sigma_u)dW_u))dW_s$$

Where we only care about $dW_tdW_t$, so finally: 
$$S_{t+dt} = S_t + \mu_t\int_{t}^{t+dt}ds+\sigma_t\int_{t}^{t+dt}dW_s+\int_{t}^{t+dt}\int_{t}^{s}(\sigma_u'\sigma_u)dW_udW_s$$

$$= S_t + \mu_t\int_{t}^{t+dt}ds + \sigma_t\int_{t}^{t+dt}dW_s + \sigma_t'\sigma\int_{t}^{t+dt}W_sdW_s$$

$$= S_t + \mu_t\int_{t}^{t+dt}ds + \sigma_t\int_{t}^{t+dt}dW_s + \sigma_t'\sigma \frac{1}{2}((\Delta W_{dt})^2 - dt)$$   

$$=S_t + \mu(S_t)dt+\sigma(S_t)\sqrt{dt}Z+\frac{1}{2}\sigma(S_t)'\sigma(S_t)dt(Z^2-1)$$

#### Milstein scheme for Black-Scholes Models

$$S_{t+dt}=S_t + rS_tdt+\sigma S_t\sqrt{dt}Z_1 + \frac{1}{2}\sigma_t^2dt(Z^2-1)$$

for log case:
$$log(S_{t+dt}) = log(S_t) + (r- \frac{\sigma^2}{2})dt + \sigma \sqrt{dt} z$$

There is no $dS_t$ item, so it's equal to Euler-Maruyama scheme.That is why we use $d(log(S_t))$ more precise than $dS_t$ case.

- Complexity: $ O(N*steps)$, with the same complexity, increase N is better than step, cause Milstein already considered the second order.

![image](https://user-images.githubusercontent.com/110284601/234842892-f0e2e65a-d603-4d31-bb95-82bfda2b288f.png)

### Longstaff-Schwartz Least Square Monte Carlo

- Assume that we know future information, so we could do the best decision for each time and the option price is close to European option.

```
def Longstaff2(S0, K, T, r, sigma, payoff, N=1000, paths=10000, order=2):
    dT = T / (N-1)
    df = np.exp(-r * dT)
    np.set_printoptions(precision=8)
    S = np.zeros((paths, N+1))
    S[:,0] = np.array([S0]*paths)
    rvs = np.random.normal(size = (paths, N))

    for i in range(N):
        S[:, i+1] = S[:, i] * np.exp((r - sigma**2/2)*dT + 
                                 sigma * np.sqrt(dT) * rvs[:, i])

    IntrinsicValues = np.maximum(S - K, 0)

    for t in range(N-1, 0, -1):
        selected_path = IntrinsicValues[:, t] >0
        regression = np.polyfit(S[selected_path, t], 
                                IntrinsicValues[selected_path, t+1] * df, order)
        expect_HV = np.polyval(regression, S[selected_path, t])
        exercises = np.zeros(len(selected_path), dtype=bool)
        exercises[selected_path] = IntrinsicValues[selected_path, t] > expect_HV
        IntrinsicValues[exercises, t+1] = -1
        IntrinsicValues[~exercises, t] = IntrinsicValues[~exercises, t+1] * df
```
print(Longstaff2(100, 100, 1, 0.1, 0.2, 'call', N=252, paths=10000, order=2)) 

13.316469702882685

Which is very close to European option.

### Techniques for variance reduction

- Antithetique Variant
- Variable Control
- Importance Sampling
- Quasi Monte Carlo

#### Antithetique variant
$$Var[C_a] = \frac{1}{4}(Var[C_X]+Var[C_Y]+2Cov(C_X,C_Y)$$
$$Var[C_a]=Var[\frac{1}{2}(h(X)+h(Y))] = \frac{\sigma^2}{2n}(1+\rho_{h(X),h(-X)})$$
where:
$$h = h_s+h_a $$
with: $$h_s(x)=\frac{1}{2}(h(x)+h(-x))$$ 
$$h_a(x)=\frac{1}{2}(h(x)-h(-x)) $$
- The more symmetry, the less effect.

![image](https://user-images.githubusercontent.com/110284601/234843464-f60acb7d-a754-4ce0-a223-ab136adbb9f2.png)

#### Variable control

$$E[h(X)] = C_c = E[h(X)+c(Z-E(Z))]$$
$$Var[h(X)+c(Z-E(Z))] = Var[h(X)]+c^2Var[Z]+2c*Cov(h(X),Z)$$

From the equation we can conduct:
$$c = -\frac{Cov(h(X),Z)}{V[Z]}$$
and $Var[C_c] = Var[C_X](1-\rho^2_{h(X),Z}) $

- The more correlation, the better

![image](https://user-images.githubusercontent.com/110284601/234843553-20d8e17a-88b8-41a6-b739-6780f84dadec.png)

#### Importance sampling
- In the part in Importance sampling, we use the technique of change of measure.
- principally, for the out-of-money option, change the float rate to minimize the payoff variance, and then divide the final result by radon nikodym derivative.

```
@MonteCarlo_Analyser_IS
def BlackScholes_MonteCarlo_IS(S0:float, K:float, T:float, sigma:float, r:float, option_type:str='call', model:str='BS',
                               N:int=1000000, seed:int=42, cores:int=8) -> float:
    np.random.seed(seed)
    rvs = np.random.normal(size=N)
    
    mu0 = np.log(K/S0)
    p = stats.norm((r - sigma**2/2)*T, sigma * np.sqrt(T))
    q = lambda mu: stats.norm((r - (sigma**2)/2 + mu)*T, sigma * np.sqrt(T))


    def var(mu=mu0, option_type=option_type):
        log_S = (r - sigma**2/2 + mu)*T + sigma * np.sqrt(T) * rvs
        S_T = S0 * np.exp(log_S)
        if option_type == 'call':
            payoff = np.maximum((S_T - K),0) * p.pdf(log_S)/q(mu).pdf(log_S)
        elif option_type == 'put':
            payoff = np.maximum((K - S_T),0) * p.pdf(log_S)/q(mu).pdf(log_S)
        return np.var(payoff)

    mu = minimize(var, mu0).x
    log_S = (r - sigma**2/2 + mu)*T + sigma * np.sqrt(T) * rvs
    S_T = S0 * np.exp(log_S)
    
    if option_type== 'call':
        payoff = np.maximum(S_T - K, 0) * np.exp(-r * T)* p.pdf(log_S) / q(mu).pdf(log_S)
    elif option_type == 'put':
        payoff = np.maximum(K - S_T, 0) * np.exp(-r * T)* p.pdf(log_S) / q(mu).pdf(log_S)
    else:
        raise ValueError("option must be call or put")
        
    return payoff

res = BlackScholes_MonteCarlo_Euler_logS(S0=100, K=140, T=1, sigma=0.2, r=0.1, option_type='call', model='BS', N=1000000, cores=8)
res_IS = BlackScholes_MonteCarlo_IS(S0=100, K=140, T=1, sigma=0.2, r=0.1, option_type='call', model='BS', N=1000000, cores=8)
```
![image](https://user-images.githubusercontent.com/110284601/234844199-dabb5d89-0c8d-4851-b1ca-a99128b13be1.png)

####  Quasi Monte Carlo
- Quasi monte carlo use non-random fixed sequence instead of pseudo-random sequence.
- In this case I use Halton sequence. The result is quite bug.

![image](https://user-images.githubusercontent.com/110284601/234844475-1860802c-087c-4e27-9f90-8e50a6d6dfff.png)

### Binomial Tree for Option Pricing
**Criteria:**
$$u*d = 1$$
$$E(S_\Delta t) = S_0 * (pu+qd)$$
$$Var(S_\Delta t) = E(S^2_\Delta t) + E^2(S_\Delta t)$$

**Result:**
$$u = exp(σ\sqrt{Δt})$$
$$d = 1/u$$
$$p =\frac{e^{rΔt}-d}{u-d}$$
$$q = 1 - p$$

![image](https://user-images.githubusercontent.com/110284601/234844559-b48f5851-916d-4692-ad03-221824a6dc56.png)



### Binomial Tree for American Option Pricing

Here, the call value equals to the European Option, while the American put is higher than European Option, which meets the reality.

![image](https://user-images.githubusercontent.com/110284601/234844607-1798b1d3-a053-4b0a-a5e3-87d115584a8f.png)


## Finite Differential Method

FDE method is faster than Monte Carlo for pricing option. And easier for American option

Explicit scheme -> Easy to apply but not stable. The probability could be negative. 

Implicit scheme -> Solve the inverse of the matrix. Hard to calculate but stable.

Crank-Nicolson scheme -> 6 points, hard to calculate. Combine Explicit and Implicit. Best precise and unconditioned stablity.

### Explicit Euler Method

![image](https://user-images.githubusercontent.com/110284601/234844732-873bc206-f07e-4d1e-99c5-06a63bed899b.png)

$$f_{i,j-1} = a_if_{i+1,j}+b_if_{i,j}+c_if_{i-1,j}$$

$$a = \frac{1}{2}\Delta t(\sigma^2i^2+ri)$$

$$b = 1-\Delta t(\sigma^2i^2+r)$$

$$c = \frac{1}{2}\Delta t(\sigma^2i^2-ri)$$ 

- Stable condition: $0<\frac{dt}{dS^2}<\frac{1}{2}$
- 
- The accuracy for Explicit Euler is dt

![image](https://user-images.githubusercontent.com/110284601/234844750-f7e4e60c-39a9-4361-ba3a-83c8d176f19e.png)

### Implicit Euler Method

![image](https://user-images.githubusercontent.com/110284601/234844783-f6eb3236-7743-4eea-8036-eb5f6fa35eb4.png)

![image](https://user-images.githubusercontent.com/110284601/234844833-a07f77b1-ee8f-49c4-b91a-05086ff18658.png)


### Crank-Nicolson

- The combination of Explicit and Implicit. The sum of the two result. And the accuracy is $(dt)^2$

$$\frac{1}{2}Explicit + \frac{1}{2}Implicit$$

$$-a_if_{i+1,j-1}+(1-b_i)f_{i,j-1}-c_if_{i-1,j-1} = a_if_{i+1,j}+(1+b_i)f_{i,j}+c_if_{i-1,j}$$

$$a_i = \frac{1}{4}\Delta t(\sigma ^2 i^2 + ri)$$

$$b_i = -\frac{1}{2}\Delta t (\sigma^2i^2+r)$$

$$c_i = \frac{1}{4}\Delta t(\sigma ^2 i^2 - ri)$$

$$M_1 f_{j-1} = M_2 f_j$$

![image](https://user-images.githubusercontent.com/110284601/234844924-c3495240-a8b8-4add-b21b-8b1bc96b501d.png)

