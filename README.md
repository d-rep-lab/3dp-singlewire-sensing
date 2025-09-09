# 3D Printed Single-Wire Sensing 

About
-----
* Source code of a computational design pipeline to generate a 3D Printed Object with capacitive touchpoints using single-wire sensing
  - The pipeline is introduced in "Computational Design and Single-Wire Sensing of 3D Printed Objects with Integrated Capacitive Touchpoints", S. Sandra Bae, Takanori Fujiwara, Ellen Yi-Luen Do, Danielle Albers Szafir, and Michael L. Rivera. Proc. SCF, 2025 (forthcoming).
  - This pipeline is extended from the one in **["A Computational Design Pipeline to Fabricate Sensing Network Physicalizations"](https://arxiv.org/abs/2308.04714)**, S. Sandra Bae, Takanori Fujiwara, Anders Ynnerman, Ellen Yi-Luen Do, Michael L. Rivera, and Danielle Albers Szafir. IEEE Transactions on Visualization and Computer Graphics, vol. 30, no. 1, pp. 913-923, Jan. 2024. [Related source code.](https://github.com/takanori-fujiwara/sensing-network)

* Links:
  * [Project page](#)
  * [Demo video](#)
  * [arXiv paper](#)

******

Content
-----
* interface_design: Web-based UI to select touchpoints and wiring connection points.


******

Interface Design Web UI  
-----

### Installation requirements
* Node.js (latest)
* Browser supporting JavaScript ES2015(ES6) and WebGL 2.0

* Note: Tested with macOS Sequoia with Google Chrome.

### Setup

* Move to `interface_design` directory; then run the below command in Terminal:

    `npm install`

## Usage

* Run

    `npx vite`

* With your browser, open the URL indicated by the above command.

* To read a different STL file, change the blow line in `script/main.js`:

    `const stlFilePath = './models/bunny.stl';`

    * Note: To avoid cross reference errors, for now, `interface_design` has its own `models` directory (i.e., this is different from `models` located at the top directory)

******
