from copy import deepcopy

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class LinearClassifier:
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        learning_rate: float = 1e-3,
        batch_size: int = 100,
        reg: float = 1e-3,
        num_iters: int = 1000,
    ):

        self.num_classes = num_classes
        self.num_features = num_features

        self.reg = reg
        self.num_iters = num_iters
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.params = dict(
            W=nn.Parameter(
                torch.randn(num_features, num_classes, dtype=torch.float)
                * np.sqrt(2 / (num_features + num_classes))
            ),
            b=nn.Parameter(torch.zeros(num_classes, dtype=torch.float)),
        )

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
            if i % 100 == 0 or i == self.num_iters - 1:

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

        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱ Assignment 3.1 ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # TODO:                                                             #
        # Compute the logits (class scores) of the linear classifier.       #
        # For a single sample x (a row vector of D features) the scores are #
        #     s = x W + b                                                   #
        # Compute them for the whole batch X at once, without a loop.       #
        #                                                                   #
        # The parameters are PyTorch tensors (torch.nn.Parameter):          #
        #     self.params["W"]  weights of shape (D, C)                     #
        #     self.params["b"]  biases of shape (C,)                        #
        # where D = number of features and C = number of classes.           #
        #                                                                   #
        # HINT 1: `@` is matrix multiplication in PyTorch:                  #
        #         (N, D) @ (D, C) -> (N, C)                                 #
        # HINT 2: Adding a tensor of shape (C,) to a tensor of shape (N, C) #
        #         adds it to every row (broadcasting).                      #
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

        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱ Assignment 3.2 ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # TODO:                                                             #
        # Predict one label per sample: the class with the highest logit.   #
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

        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱ Assignment 3.3 ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # TODO:                                                             #
        # Compute the cross-entropy loss with L2 regularization:            #
        #     loss = CE(logits, y) + reg * (sum(W ** 2) + sum(b ** 2))      #
        # where CE is the mean cross-entropy over the batch and reg is      #
        # self.reg. Both parameters are regularized:                        #
        #     W = self.params["W"]  of shape (D, C)                         #
        #     b = self.params["b"]  of shape (C,)                           #
        #                                                                   #
        # Shapes: X is (N, D), y is (N,) with integer labels 0 ... C-1.     #
        #                                                                   #
        # HINT 1: torch.nn.CrossEntropyLoss()(logits, y) takes the raw      #
        #         logits of shape (N, C), not probabilities. It applies     #
        #         the softmax itself and averages over the batch.           #
        # HINT 2: torch.sum(W ** 2) is the sum of squares of all entries.   #
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

        W = self.params["W"]  # weights of shape (D, C)
        b = self.params["b"]  # biases of shape (C,)

        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱ Assignment 3.4 ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # TODO:                                                             #
        # Take one gradient-descent step for both parameters:               #
        #     W = W - learning_rate * dL/dW                                 #
        #     b = b - learning_rate * dL/db                                 #
        #                                                                   #
        # loss.backward() has already computed the gradients and stored     #
        # them next to the parameters:                                      #
        #     W.grad  has the same shape as W, (D, C)                       #
        #     b.grad  has the same shape as b, (C,)                         #
        #                                                                   #
        # HINT 1: Use self.learning_rate for the learning rate.             #
        # HINT 2: Write the new values into `.data`, for example            #
        #             W.data = W.data - ...                                 #
        #         Writing `W = W - ...` would only create a new local       #
        #         variable and leave the model unchanged. `.data` also      #
        #         keeps the update out of the autograd graph.               #
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
