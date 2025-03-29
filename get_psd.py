1. Train two Gaussian processes: ExactGP and Non-stationary GP. 
2. Sample a grid of points from these processes.
3. Data preprocessing: detrend and normalize.
4. Conduct the fourier analysis

Expected output:

- Sampled points used to train the Gaussian process
- The resulting gaussian process from sample points (2) 
- The power density graphs (2)
- The dominant frequencies