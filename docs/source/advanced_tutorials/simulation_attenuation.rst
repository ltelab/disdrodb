-------------------------------------------------
Multi-Frequency Radar Simulations with Dask
-------------------------------------------------

This tutorial demonstrates how to compute radar variables at multiple frequencies
using parallel processing with Dask. This approach enables efficient computation
of radar observables across a wide frequency range for large datasets.

**Prerequisites**

- ``pytmatrix`` package installed (see :ref:`pytmatrix installation <pytmatrix_installation>`)
- Sufficient memory for parallel processing (adjust chunk sizes accordingly)
- DISDRODB L2E product available for the station

**Overview**

The workflow consists of:

1. Initialize Dask cluster for parallel processing
2. Load DISDRODB L2E product with lazy loading
3. Configure data chunking for optimal memory usage
4. Compute radar variables at multiple frequencies
5. Analyze and visualize results

**Step 1: Initialize Dask Cluster**

Dask enables parallel processing and provides a dashboard for monitoring computations.

.. code-block:: python

    import numpy as np
    import xarray as xr
    import disdrodb
    from disdrodb.utils.dask import initialize_dask_cluster

    # Initialize Dask cluster
    # - Dashboard available at http://localhost:8787/status
    # - Monitor memory usage and task progress
    initialize_dask_cluster()

**Step 2: Load L2E Product**

Load the L2E product with lazy loading enabled (``parallel=True``, ``chunks="auto"``).
This delays computation until explicitly requested.
This code snippet assumes you have already processed the L2E product for the station.
If not, see the :ref:`Quick Start <quick_start>` guide for processing instructions.

.. code-block:: python

    # Specify station
    data_source = "EPFL"
    campaign_name = "HYMEX_LTE_SOP3"
    station_name = "10"

    # Load dataset with lazy loading
    ds = disdrodb.open_dataset(
        product="L2E",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        parallel=True,
        chunks="auto",
        temporal_resolution="1MIN",
    )

    # Select single velocity method (if multiple available)
    ds = ds.sel(velocity_method="theoretical_velocity")

    # Load data into memory
    ds = ds.compute()

.. note::

   The initial ``ds.compute()`` loads the L2E product into memory. For very large datasets,
   you may want to skip this step and work with lazy arrays throughout.

**Step 3: Configure Data Chunking**

Rechunk the dataset to optimize parallel processing. Chunk size determines the trade-off
between parallelization and memory usage.

.. code-block:: python

    # Rechunk for parallel radar computations
    # - Larger chunks = more memory per worker
    # - Smaller chunks = more parallelization overhead
    # - 5000 timesteps ≈ 12 GB RAM per worker
    ds = ds.chunk({"time": 5000})

.. tip::

   **Memory Considerations**

   - Monitor memory usage on Dask dashboard during processing
   - Reduce chunk size if workers run out of memory
   - Increase chunk size for faster processing on high-memory machines

**Step 4: Compute Multi-Frequency Radar Variables**

Compute radar observables across multiple frequencies. T-matrix lookup tables (LUTs)
are computed once and cached for subsequent runs.

.. code-block:: python

    # Define frequency range (GHz)
    frequencies = np.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]).tolist()

    # Compute radar variables
    # - First run: LUTs computed for all frequencies (several minutes)
    # - Subsequent runs: Cached LUTs used (much faster)
    # - Monitor progress at http://localhost:8787/status
    ds_radar = disdrodb.generate_l2_radar(
        ds=ds,
        frequency=frequencies,
        num_points=1024,  # T-matrix integration points
        diameter_max=10,  # Maximum diameter (mm)
        canting_angle_std=7,  # Canting angle std dev (degrees)
        axis_ratio_model="Thurai2007",
        permittivity_model="Turner2016",
        water_temperature=10,  # Water temperature (°C)
        elevation_angle=0,  # Elevation angle (degrees)
        parallel=True,
    )

For details on radar simulation parameters, see :ref:`Radar Variable Simulations <disdrodb_radar>`
and :ref:`Radar Simulation Options <products_configuration>`.

**Step 5: Analyze Results**

Explore the computed radar variables and their frequency dependence.

.. code-block:: python

    # Available radar variables include:
    # - AH, AV: Specific attenuation at H and V polarization (dB/km)
    # - ZH, ZV: Reflectivity at H and V polarization (dBZ)
    # - ZDR: Differential reflectivity (dB)
    # - KDP: Specific differential phase (deg/km)
    # - RHOHV: Copolar correlation coefficient

    # Inspect attenuation variables
    print(ds_radar["AH"])
    print(ds_radar["AV"])

    # Select timestep with highest rain rate
    idx = ds["R"].compute().argmax().item()

    # Or select specific timestep
    idx = 4
    ds_radar.isel(time=idx)

**Step 6: Visualize Frequency-Dependent Attenuation**

Plot how attenuation varies with frequency for a single timestep.

.. code-block:: python

    import matplotlib.pyplot as plt

    # Define timestep index
    idx = 4

    # Plot vertical polarization attenuation
    da_av = ds_radar.isel(time=idx)["AV"]
    da_av.plot(x="frequency", marker="o")
    plt.title(f"Vertical Polarization Attenuation (R={ds.isel(time=idx)['R'].values:.2f} mm/h)")
    plt.ylabel("Attenuation (dB/km)")
    plt.xlabel("Frequency (GHz)")
    plt.grid(True)
    plt.show()

    # Plot horizontal polarization attenuation
    da_ah = ds_radar.isel(time=idx)["AH"]
    da_ah.plot(x="frequency", marker="o")
    plt.title(f"Horizontal Polarization Attenuation (R={ds.isel(time=idx)['R'].values:.2f} mm/h)")
    plt.ylabel("Attenuation (dB/km)")
    plt.xlabel("Frequency (GHz)")
    plt.grid(True)
    plt.show()

**Step 7: Compare Polarizations**

Analyze differential attenuation across frequencies.

.. code-block:: python

    # Compute differential attenuation
    ds_radar["A_diff"] = ds_radar["AH"] - ds_radar["AV"]

    # Plot differential attenuation
    da_diff = ds_radar.isel(time=idx)["A_diff"]
    da_diff.plot(x="frequency", marker="o", color="purple")
    plt.title("Differential Attenuation (AH - AV)")
    plt.ylabel("Differential Attenuation (dB/km)")
    plt.xlabel("Frequency (GHz)")
    plt.grid(True)
    plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    plt.show()

**Performance Optimization Tips**

1. **LUT Caching**: First run computes and caches T-matrix LUTs. Subsequent runs with
   the same parameters are much faster.

2. **Chunk Size Tuning**: Adjust ``chunks={"time": N}`` based on available memory:

   - Small datasets: Use larger chunks or ``chunks=None`` for in-memory processing
   - Large datasets: Reduce chunk size if memory errors occur

3. **Frequency Selection**: Computing more frequencies increases total time. Process frequencies in batches if needed.

4. **Parallel Processing**: Ensure sufficient CPU cores and memory per worker:

   - Monitor Dask dashboard for bottlenecks
   - Adjust ``DASK_NUM_WORKERS`` environment variable if needed.

5. **Storage**: Save computed radar variables to avoid recomputation:

   .. code-block:: python

       # Save results
       ds_radar.to_netcdf("radar_multifreq.nc")

       # Load saved results
       ds_radar = xr.open_dataset("radar_multifreq.nc")
