This repository implements a two-player secure multi-party computation (MPC) framework using a custom-built local socket network. The framework enables two parties (Alice and Bob) to jointly compute a function while maintaining data privacy.

Features:
- **Two-Party Computation (2PC):** Secure computation between Alice and Bob.
- **Custom Low-Level Network:** Implements a local socket-based network using IPv4 and SOCK_STREAM for communication.
- **Pre-Shared Cryptographic Material:** Uses pre-generated Beaver’s triples and EdaBits for efficient MPC execution.
- **Modular Design:** Organized into separate modules for networking, secure computation protocols, and function evaluation.
