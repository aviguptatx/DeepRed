<p align="center">
  <a href="https://github.com/aviguptatx/SecretHitlerAI">
  </a>

  <h3 align="center">Secret Hitler AI</h3>

  <p align="center">
    LSTM-RNN for Hidden Role Prediction in Secret Hitler using PyTorch
    <br />
    <br />
    <a href="https://github.com/aviguptatx/SecretHitlerAI/issues">Report Bug</a>
    ·
    <a href="https://github.com/aviguptatx/SecretHitlerAI/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)



<!-- ABOUT THE PROJECT -->
## About The Project

Long Short-Term Memory Recurrent Neural Network for hidden role prediction in 7-player [Secret Hitler](https://en.wikipedia.org/wiki/Secret_Hitler) using [PyTorch](https://pytorch.org/). The program takes in plaintext game data, which is eventually converted to a one-hot encoded input layer for the network. The network returns a Tensor containing probabilities that each player is a fascist.

Secret Hitler is a game of lies and deception, and more formally, it is a game of both imcomplete and imperfect information. Thus, predicting the role of a given player in Secret Hitler is actually quite difficult and the network required lots of fine tuning and iteration to get to the level of accuracy it currently achieves.

The network was trained on about 20,000 games worth of data, using [Google Colab's](https://colab.research.google.com/notebooks/intro.ipynb) free cloud GPUs. On a validation set of about 5,000 games, the network achieved a mean error of 1.620, or about .231 per player. A player that guesses randomly (but knows their own role) achieves an error of 3.000, or about .429 per player. The network's predictions fare quite well against novice players who are at or under 1600 elo on [secrethitler.io](https://secrethitler.io/).

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

In addition to [Python 3.8.2](https://www.python.org/downloads/release/python-382/), the following was used in this program:
* NumPy v1.18.4
```sh
pip install numpy==1.18.4
```

* PyTorch v1.5.0
```sh
pip install torch==1.5.0 torchvision==0.6.0
```

### Installation
 
Just clone the repo. As long as the prerequisites have been filled, the program should be ready to run.
```sh
git clone https://github.com/aviguptatx/SecretHitlerAI.git
```

<!-- USAGE EXAMPLES -->
## Usage

To use the network on a custom game, simply edit `game_in.txt`, following the instructions given in the file. After `game_in.txt` has been filled with plaintext game data in the correct format, simply run `RNN.py`, and the program will print its prediction.

<!-- CONTRIBUTING -->
## Contributing
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

Avinash Gupta - aviguptatx@gmail.com

Varun Gorti - varungorti@gmail.com

Project Link: [https://github.com/aviguptatx/SecretHitlerAI](https://github.com/aviguptatx/SecretHitlerAI)

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
Thank you to [secrethitler.io](https://secrethitler.io/) for providing game data to train on. This project would not be possible without the awesome folks over there.
