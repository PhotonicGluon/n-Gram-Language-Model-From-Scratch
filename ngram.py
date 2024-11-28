import gzip
import random
import re
from math import exp


class TokenNode:
    def __init__(self, token: str | None = None, count: int = 1):
        """
        Constructor for TokenNode

        Args:
            token: The token stored in this node. If None, "<ROOT>" is used.
            count: The count of this token. Defaults to 1.
        """

        self._token = token if token is not None else "<ROOT>"
        self._count = count

        self._children: list[TokenNode] = []

    # Properties
    @property
    def token(self) -> str:
        return self._token

    @property
    def count(self) -> int:
        return self._count

    # Magic methods
    def __repr__(self) -> str:
        return f"TokenNode(token={self.token}, count={self.count})"

    def __getitem__(self, idx: str | list[str] | tuple[str]) -> "TokenNode":
        """
        Allows indexing into the TokenNode tree to access its children.

        Parameters:
            idx: A single token or tuple of tokens to index into the tree.

        Returns:
            The TokenNode at the given index.

        Raises:
            IndexError: If the index is invalid.
        """

        if isinstance(idx, str):
            for child in self._children:
                if child.token == idx:
                    return child
            raise IndexError(f"Invalid index: {idx}")

        result = self
        for i in idx:
            result = result[i]
        return result

    # Helper methods
    def _print_tree(self, depth: int = 0):
        """
        Prints the tree rooted at this node in a human-readable format.

        Recursively prints the tree, with each level of nesting indented by 2 spaces.

        Parameters:
            depth: The current depth of the tree. Defaults to 0.
        """

        print(f"{'  ' * depth} {self.token} ({self.count})")
        for child in self._children:
            child._print_tree(depth=depth + 1)

    def _add_descendant(self, ngram: tuple[str]) -> None:
        """
        Recursively adds a given ngram to the tree.

        Parameters:
            ngram: The ngram to add to the tree.
        """

        # Increment self's count
        self._count += 1

        # If empty, nothing to do
        if len(ngram) == 0:
            return

        # Find child to add this to
        for child in self._children:
            if child.token == ngram[0]:
                # Continue adding to child
                child._add_descendant(ngram[1:])
                return

        # If reached here, can't find a child to add this to; create a new one
        child = TokenNode(ngram[0], count=0)  # We will add count later
        self._children.append(child)
        child._add_descendant(ngram[1:])

    def _prune_tree(self, threshold: float):
        """
        Recursively prunes the tree rooted at this node.

        Removes all nodes that have a count proportion of less than threshold
        compared to their parent node.

        Parameters:
            threshold: The minimum proportion of a node's count for it to be kept.
        """

        # If no children, no need to prune
        if len(self._children) == 0:
            return

        # Find children whose count proportion is less than threshold
        for child in reversed(self._children):
            if child._count / self._count < threshold:
                self._count -= child._count
                self._children.remove(child)
            else:
                child._prune_tree(threshold)

    # Public methods
    def print_tree(self):
        """
        Prints the tree rooted at this node in a human-readable format.
        """

        self._print_tree()

    def serialize(self) -> str:
        """
        Serializes the TokenNode and its descendants into a string representation.

        Each node is represented as a line in the format "token count", with child nodes
        indented by one space per level of depth. The serialization is done recursively
        for all children.

        Returns:
            A string representing the serialized tree rooted at this node.
        """

        output = f"{self.token} {self.count}\n"
        for child in self._children:
            child_as_str = child.serialize()
            lines = child_as_str.split("\n")
            for line in lines:
                output += f" {line}\n"

        return output[:-1]  # Remove last newline

    @classmethod
    def deserialize(cls, serialized: str) -> "TokenNode":
        """
        Deserializes a string representation of a TokenNode into a TokenNode object.

        The serialized string is expected to have each node represented as a line in the format
        "token count", with child nodes indented by one space per level of depth. This method
        reconstructs the tree structure from the serialized format.

        Args:
            serialized: A string representing the serialized tree of TokenNodes.

        Returns:
            The root TokenNode of the deserialized tree.
        """

        lines = serialized.split("\n")

        # Get the token and count
        token = lines[0].split(" ")[0]
        count = int(lines[0].split(" ")[1])

        # Unindent the children part
        children_part = lines[1:]
        for i in range(len(children_part)):
            children_part[i] = children_part[i][1:]

        # Get the children
        children_indices = []
        for i in range(len(children_part)):
            if not children_part[i][0].isspace():
                children_indices.append(i)
        children_indices.append(len(children_part))
        children = [children_part[i:j] for i, j in zip(children_indices[:-1], children_indices[1:])]
        children = ["\n".join(child) for child in children]

        # Create the node
        node = cls(token, count)
        node._children = [cls.deserialize(child) for child in children]
        return node

    def add_descendant(self, ngram: tuple[str]) -> None:
        """
        Recursively adds a given ngram to the tree.

        Parameters:
            ngram: The ngram to add to the tree.
        """

        if len(ngram) == 0:
            return

        self._add_descendant(ngram)

    def prune_tree(self, threshold: float = 0.0025):
        """
        Prunes the tree by removing nodes with a relative frequency below the given threshold.

        Parameters:
            threshold: The minimum relative frequency for a node to be kept in the tree.
        """

        # Ensure threshold is between 0 and 1
        if threshold < 0 or threshold > 1:
            raise ValueError("Threshold must be between 0 and 1")

        # Prune the tree
        self._prune_tree(threshold)


class NGramModel:
    def __init__(self, n: int):
        """
        Constructor for NGramModel.

        Parameters:
            n: The order of the model. This is the length of n-grams that the model will store.
        """

        self._n = n
        self._root = TokenNode(count=0)

    # Properties
    @property
    def size(self) -> int:
        """
        Returns the number of n-grams stored in the model.
        """

        return self._root.count

    # Magic methods
    def __repr__(self) -> str:
        return f"NGramModel(n={self._n})"

    def __str__(self) -> str:
        return repr(self)

    # Helper methods
    def _get_next_word_probabilities(
        self, ngram: tuple[str], temperature: float = 1.0, pattern: str = r".*", excluded: list[str] | None = None
    ) -> dict[str, float]:
        """
        Calculates the probabilities of next words following a given n-gram using a temperature-
        regulated softmax.

        Parameters:
            ngram: A tuple of strings representing the n-gram to find the next word probabilities
                for.
            temperature: A float representing the temperature for softmax adjustment. Defaults to
                1.0.
            pattern: A regular expression pattern to filter tokens. Defaults to match any string.
            excluded: A list of strings to exclude from the probabilities. Defaults to None.

        Returns:
            A dictionary mapping each token to its probability of being the next word, adjusted by
            temperature.
        """

        if excluded is None:
            excluded = []

        # Get the node for this ngram
        node = self._root[ngram]

        # Get the relevant children
        children = [child for child in node._children if re.match(pattern, child.token) and child.token not in excluded]
        num_relevant = sum(child.count for child in children)

        # Get the raw probabilities of each of the children
        child_raw_probabilities = {child.token: child.count / num_relevant for child in children}

        # Apply temperature-regulated softmax
        child_adjusted_probabilities = {}
        softmax_sum = 0
        for token, p in child_raw_probabilities.items():
            adj_p = exp(p / temperature)
            child_adjusted_probabilities[token] = adj_p
            softmax_sum += adj_p
        child_adjusted_probabilities = {
            token: adj_p / softmax_sum for token, adj_p in child_adjusted_probabilities.items()
        }

        return child_adjusted_probabilities

    # Public methods
    def print_tree(self):
        """
        Prints the tree rooted at the model's root node in a human-readable format.

        The tree is printed with each level of nesting indented by 2 spaces.

        This is a debugging method and should not be used in production code.
        """

        self._root.print_tree()

    def serialize(self) -> str:
        """
        Serializes the model to a string.
        """

        return f"{self._n}\n{self._root.serialize()}"

    @classmethod
    def deserialize(cls, serialized: str) -> "NGramModel":
        """
        Deserializes a string representation of an NGramModel into an instance of the class.

        Parameters:
            serialized: A string containing the serialized representation of an NGramModel.

        Returns:
            An instance of NGramModel constructed from the serialized string.
        """

        n = int(serialized.split("\n")[0])
        instance = cls(n)
        instance._root = TokenNode.deserialize("\n".join(serialized.split("\n")[1:]))
        return instance

    def save(self, filename: str):
        """
        Saves the serialized model to a file.

        Parameters:
            filename: The name of the file where the serialized model will be saved.
        """

        # Get the serialized form
        serialized = self.serialize()

        # Save it to file
        with gzip.open(filename, "w") as f:
            f.write(serialized.encode("UTF-8"))

    @classmethod
    def load(cls, filename: str) -> "NGramModel":
        """
        Loads an NGramModel instance from a file.

        Parameters:
            filename: The name of the file where the serialized model is saved.

        Returns:
            An instance of NGramModel constructed from the serialized string loaded from the file.
        """

        # Load the serialized form
        with gzip.open(filename, "r") as f:
            serialized = f.read().decode("UTF-8")

        # Deserialize it
        return cls.deserialize(serialized)

    def add_ngram(self, ngram: tuple[str]) -> None:
        """
        Adds a given ngram to the model.

        Parameters:
            ngram: The ngram to add to the model.

        Raises:
            ValueError: If the ngram is invalid.
        """

        if len(ngram) != self._n:
            raise ValueError(f"Invalid ngram: {ngram}")

        self._root.add_descendant(ngram)

    def prune(self, threshold: float = 0.0025):
        """
        Prunes the n-gram model by removing low-frequency nodes.

        Parameters:
            threshold: The minimum relative frequency for a node to be retained in the model.
        """

        self._root.prune_tree(threshold=threshold)

    def generate_text(
        self,
        start_ngram: tuple[str],
        num_to_generate: int = 1,
        seed: int | None = None,
        temperature: float = 1.0,
        patterns: list[str] | None = None,
        exclusions: list[str] | None = None,
    ) -> list[str]:
        """
        Generates a sequence of words based on the n-gram model.

        Parameters:
            start_ngram: A tuple representing the starting n-gram for text generation.
            num_to_generate: The number of words to generate.
            seed: An optional seed for random number generation to ensure reproducibility.
            temperature: A float value to control randomness in generation; higher values increase randomness.
            patterns: A list of regex patterns that the generated words must match.
            exclusions: A list of lists containing words to exclude at each generation step.

        Returns:
            A list of generated words, starting from the provided n-gram.

        Raises:
            ValueError: If the number of patterns or exclusions does not match `num_to_generate` or if text generation fails.
        """

        if not isinstance(start_ngram, tuple):
            raise ValueError(f"Invalid start ngram: {start_ngram} (not a tuple)")

        if patterns is None:
            patterns = [r".*"] * num_to_generate
        if len(patterns) != num_to_generate:
            raise ValueError(f"Invalid number of patterns: {patterns} (should be {num_to_generate})")

        if exclusions is None:
            exclusions = [[] for _ in range(num_to_generate)]
        if len(exclusions) != num_to_generate:
            raise ValueError(f"Invalid number of exclusions: {exclusions} (should be {num_to_generate})")

        r = random.Random(seed)
        generated_text = list(start_ngram)
        i = 0
        while 0 <= i < num_to_generate:
            # Try to get the probabilities of the next word
            try:
                probabilities = self._get_next_word_probabilities(
                    generated_text[i : i + self._n],
                    temperature=temperature,
                    pattern=patterns[i],
                    excluded=exclusions[i],
                )

                if len(probabilities) == 0:
                    raise IndexError
            except IndexError:
                # If there are no probabilities, go back one step
                failed_word = generated_text.pop()
                exclusions[i - 1].append(failed_word)
                i -= 1
                continue

            # Sample the next word
            next_word = r.choices(list(probabilities.keys()), weights=list(probabilities.values()), k=1)[0]
            generated_text.append(next_word)
            i += 1

        if i == -1:
            raise ValueError("Failed to generate text")

        return generated_text[-num_to_generate:]
