from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class MLPClassifier:
    def __init__(
        self,
        num_features: int,
        hidden_dim_1: int,
        hidden_dim_2: int,
        num_classes: int,
        activation: str = "relu",
        learning_rate: float = 1e-3,
        reg: float = 1e-6,
        batch_size: int = 100,
        num_iters: int = 1000,
    ):

        self.params = dict(
            W1=nn.Parameter(
                torch.randn(num_features, hidden_dim_1, dtype=torch.float)
                * np.sqrt(2 / (num_features + num_classes))
            ),
            b1=nn.Parameter(torch.zeros(hidden_dim_1, dtype=torch.float)),
            W2=nn.Parameter(
                torch.randn(hidden_dim_1, hidden_dim_2, dtype=torch.float)
                * np.sqrt(2 / (num_features + num_classes))
            ),
            b2=nn.Parameter(torch.zeros(hidden_dim_2, dtype=torch.float)),
            W3=nn.Parameter(
                torch.randn(hidden_dim_2, num_classes, dtype=torch.float)
                * np.sqrt(2 / (num_features + num_classes))
            ),
            b3=nn.Parameter(torch.zeros(num_classes, dtype=torch.float)),
        )

        self.reg = reg
        self.num_iters = num_iters
        self.activation = activation
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        self.activation_func = None

        if activation == "relu":
            self.activation_func = nn.ReLU()
        elif activation == "sigmoid":
            self.activation_func = nn.Sigmoid()
        elif activation == "tanh":
            self.activation_func = nn.Tanh()
        elif activation == "identity":
            # No non-linearity at all, f(x) = x
            self.activation_func = nn.Identity()

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        verbose: bool = True,
    ) -> tuple:

        # Initialize the best validation accuracy and the best parameters
        best_val_acc = 0
        best_params = dict()

        # Initialize the loss and accuracy history
        loss_history = dict(train=dict(), val=dict())
        acc_history = dict(train=dict(), val=dict())

        # Training loop
        for i in tqdm(range(self.num_iters), desc="Training", disable=not verbose):

            # Select a random batch of data
            batch_indices = torch.randint(0, X_train.shape[0], (self.batch_size,))
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Zero the gradients
            self._zero_gradients()

            # Compute the loss and backpropagate
            train_loss = self.loss(X_batch, y_batch)
            train_loss.backward(retain_graph=True)
            self._update_weights()

            # Save the training loss
            loss_history["train"][i] = train_loss.data

            # Every 500 iterations, compute the validation loss and accuracy
            if i % 500 == 0 or i == self.num_iters - 1:

                # Compute the validation loss
                with torch.no_grad():
                    val_loss = self.loss(X_val, y_val)
                loss_history["val"][i] = val_loss.data

                # Predict the labels for the training and validation data
                y_pred_train = self.predict(X_train)
                y_pred_val = self.predict(X_val)

                # Compute the training and validation accuracy from the predicted labels
                acc_history["train"][i] = accuracy_score(y_train, y_pred_train)
                acc_history["val"][i] = accuracy_score(y_val, y_pred_val)

                # If the current validation accuracy is the best so far, save the parameters
                if acc_history["val"][i] > best_val_acc:
                    best_val_acc = acc_history["val"][i]
                    best_params = deepcopy(self.params)

        # Update the parameters with the best ones
        self.params = best_params

        return loss_history, acc_history

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the logits of the model.

        Args:
            X: Input data of shape (N, D)

        Returns:
            logits: The logits of the model. Tensor of shape (N, C)

        """

        logits = torch.zeros((X.shape[0], self.num_classes))

        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱ Assignment 4.1 ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # TODO:                                                             #
        # Compute the logits (class scores) of the network with two hidden  #
        # layers:                                                           #
        #     h1     = f(X  @ W1 + b1)                                      #
        #     h2     = f(h1 @ W2 + b2)                                      #
        #     logits =    h2 @ W3 + b3                                      #
        # where f is the activation function self.activation_func (ReLU,    #
        # sigmoid or tanh). Compute it for the whole batch X at once.       #
        #                                                                   #
        # The parameters are PyTorch tensors (torch.nn.Parameter):          #
        #     self.params["W1"]  weights of shape (D, H1)                   #
        #     self.params["b1"]  biases of shape (H1,)                      #
        #     self.params["W2"]  weights of shape (H1, H2)                  #
        #     self.params["b2"]  biases of shape (H2,)                      #
        #     self.params["W3"]  weights of shape (H2, C)                   #
        #     self.params["b3"]  biases of shape (C,)                       #
        # where D = number of features, H1 and H2 = sizes of the two        #
        # hidden layers and C = number of classes.                          #
        #                                                                   #
        # The shapes along the way:                                         #
        #     X (N, D) -> h1 (N, H1) -> h2 (N, H2) -> logits (N, C)         #
        #                                                                   #
        # HINT 1: Every layer is the linear classifier from Assignment 3.1: #
        #         `@` is matrix multiplication, the bias is broadcast.      #
        # HINT 2: self.activation_func is a function, call it as            #
        #         self.activation_func(tensor). It is applied to every      #
        #         element separately, so the shape stays the same.          #
        # HINT 3: No activation after the last layer. The loss expects      #
        #         the raw scores (logits).                                  #
        #                                                                   #
        # Good luck!                                                        #
        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # 🌀 INCEPTION 🌀 (Your code begins its journey here. 🚀 Do not delete this line.)
        #
        #                    ╔═══════════════════════╗
        #                    ║                       ║
        #                    ║       YOUR CODE       ║
        #                    ║                       ║
        #                    ╚═══════════════════════╝
        #

        # 🌀 TERMINATION 🌀 (Your code reaches its end. 🏁 Do not delete this line.)

        return logits

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict the labels of the data.

        Args:
            X (torch.Tensor): Input data of shape (N, D)

        Returns:
            y_pred (torch.Tensor): The predicted labels of the data. Array of shape (N,)
        """

        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱ Assignment 4.2 ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # TODO:                                                             #
        # Predict one label per sample: the class with the highest logit.   #
        # This is the same as Assignment 3.2.                               #
        #                                                                   #
        # Shapes: X is (N, D), the logits from self.forward(X) are (N, C)   #
        # and y_pred must be a PyTorch tensor of shape (N,).                #
        #                                                                   #
        # HINT 1: Use self.forward to get the logits.                       #
        # HINT 2: torch.argmax(tensor, dim=...) returns the index of the    #
        #         largest value along one dimension. Which dimension        #
        #         holds the classes?                                        #
        # HINT 3: Predictions need no gradients, so you can wrap the code   #
        #         in `with torch.no_grad():`.                               #
        #                                                                   #
        # Good luck!                                                        #
        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # 🌀 INCEPTION 🌀 (Your code begins its journey here. 🚀 Do not delete this line.)
        #
        #                    ╔═══════════════════════╗
        #                    ║                       ║
        #                    ║       YOUR CODE       ║
        #                    ║                       ║
        #                    ╚═══════════════════════╝
        #

        # 🌀 TERMINATION 🌀 (Your code reaches its end. 🏁 Do not delete this line.)

        return y_pred

    def loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the loss of the model.

        Args:
            X (torch.Tensor): Input data of shape (N, D)
            y (torch.Tensor): Labels of shape (N,)

        Returns:
            torch.Tensor: The loss of the model
        """

        loss = torch.tensor([0.0], requires_grad=True)

        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱ Assignment 4.3 ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # TODO:                                                             #
        # Compute the cross-entropy loss with L2 regularization of ALL six  #
        # parameters (weights and biases):                                  #
        #     loss = CE(logits, y) + reg * R                                #
        #     R    = sum(W1 ** 2) + sum(b1 ** 2) + sum(W2 ** 2)             #
        #          + sum(b2 ** 2) + sum(W3 ** 2) + sum(b3 ** 2)             #
        # where CE is the mean cross-entropy over the batch and reg is      #
        # self.reg. The shapes of the parameters are listed in 4.1.         #
        #                                                                   #
        # Shapes: X is (N, D), y is (N,) with integer labels 0 ... C-1.     #
        #                                                                   #
        # HINT 1: torch.nn.CrossEntropyLoss()(logits, y) takes the raw      #
        #         logits of shape (N, C), not probabilities. It applies     #
        #         the softmax itself and averages over the batch.           #
        # HINT 2: self.params is a dictionary, so                           #
        #             for param in self.params.values():                    #
        #         visits all six parameters. No need to write six terms.    #
        # HINT 3: Build the loss from PyTorch operations only (no numpy,    #
        #         no .item()), so that loss.backward() can compute          #
        #         the gradients.                                            #
        #                                                                   #
        # Good luck!                                                        #
        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # 🌀 INCEPTION 🌀 (Your code begins its journey here. 🚀 Do not delete this line.)
        #
        #                    ╔═══════════════════════╗
        #                    ║                       ║
        #                    ║       YOUR CODE       ║
        #                    ║                       ║
        #                    ╚═══════════════════════╝
        #

        # 🌀 TERMINATION 🌀 (Your code reaches its end. 🏁 Do not delete this line.)

        return loss

    def _update_weights(self):
        """Update the weights of the model using the gradient descent."""

        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱ Assignment 4.4 ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # TODO:                                                             #
        # Take one gradient-descent step for all six parameters             #
        # W1, b1, W2, b2, W3, b3:                                           #
        #     param = param - learning_rate * dL/dparam                     #
        # This is the step from Assignment 3.4, applied to every parameter. #
        #                                                                   #
        # loss.backward() has already computed the gradients and stored     #
        # them next to the parameters: param.grad has the same shape as     #
        # param, e.g. self.params["W1"].grad is (D, H1).                    #
        #                                                                   #
        # HINT 1: Use self.learning_rate for the learning rate.             #
        # HINT 2: Loop over the parameters with                             #
        #             for param in self.params.values():                    #
        # HINT 3: Write the new values into `.data`, for example            #
        #             param.data = param.data - ...                         #
        #         Writing `param = param - ...` would only create a new     #
        #         local variable and leave the model unchanged.             #
        #                                                                   #
        # Good luck!                                                        #
        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # 🌀 INCEPTION 🌀 (Your code begins its journey here. 🚀 Do not delete this line.)
        #
        #                    ╔═══════════════════════╗
        #                    ║                       ║
        #                    ║       YOUR CODE       ║
        #                    ║                       ║
        #                    ╚═══════════════════════╝
        #

        # 🌀 TERMINATION 🌀 (Your code reaches its end. 🏁 Do not delete this line.)

    def _zero_gradients(self):
        """Zero the gradients of the model parameters."""
        for name in self.params.keys():
            if self.params[name].grad is not None:
                self.params[name].grad.zero_()
