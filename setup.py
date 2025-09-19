from distutils.core import setup

setup(
    name="sw-sensing",
    version=0.1,
    package_dir={"": "."},
    install_requires=[
        "lcapy==1.21",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "scikit-learn==1.4.1.post1",
        "vtk==9.3.0",
        "pyvista==0.43.3",
        "sensing-network",
    ],
    py_modules=[
        "sw_sensing.geom_utils",
        "sw_sensing.path_finding",
        "sw_sensing.geometry",
        "sw_sensing.single_wiring_optimization",
    ],
)
