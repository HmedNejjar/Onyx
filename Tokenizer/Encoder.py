from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE as HF_BPE
from tokenizers.trainers import BpeTrainer

from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE as HF_BPE
from tokenizers.trainers import BpeTrainer


class BPE:
    """
    Byte Pair Encoding (BPE) tokenizer wrapper using Hugging Face's tokenizers library.
    
    Attributes:
        tokenizer (Tokenizer): The underlying Hugging Face tokenizer instance.
        trainer (BpeTrainer): Configured trainer for BPE algorithm.
    """
    
    def __init__(self, vocab_size: int = 30_000, min_frequency: int = 3) -> None:
        """
        Initialize the BPE tokenizer with specified parameters.
        
        Args:
            vocab_size (int): Maximum size of the vocabulary to learn.
                             Larger vocabularies capture more word pieces but increase model size.
                             Defaults to 30,000.
            min_frequency (int): Minimum frequency a character pair must have to be merged.
                                Higher values result in fewer merges and smaller vocabularies.
                                Defaults to 3.
        """
        # Initialize the underlying Hugging Face tokenizer with BPE model
        # HF_BPE implements the original Byte Pair Encoding algorithm from Sennrich et al.
        self.tokenizer = Tokenizer(HF_BPE())
        
        # Configure the trainer with user-specified parameters
        # vocab_size=vocab_size - 5, leaving IDs for special character, added later
        self.trainer = BpeTrainer(vocab_size=vocab_size - 5, min_frequency=min_frequency, show_progress=True)
    
    def train(self, file: str | Path) -> None:
        """
        Train the BPE tokenizer on a text corpus file
        
        Args:
            file (str | Path): Path to the training corpus text file.
                              Should be a plain text file (UTF-8 encoded).
        
        Raises:
            FileNotFoundError: If the specified file does not exist.
            IOError: If the file cannot be read.
        """
        # Define special tokens that will be added to the vocabulary
        # These tokens are essential for many NLP tasks and transformer models
        special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
        
        print(f"Training BPE on '{file}'.")
        
        # Train the tokenizer on the provided file
        self.tokenizer.train([str(file)], self.trainer)
        
        # Add special tokens to the vocabulary after training
        # This ensures they get reserved vocabulary IDs
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Display the final vocabulary size for verification
        print(f"Vocab size: {self.tokenizer.get_vocab_size()}")
    
    def save(self, path: str | Path) -> None:
        """
        Save the trained tokenizer to a JSON file for later use.
        
        Args:
            path (str | Path): Destination path for the tokenizer file in JSON format.
        """

        self.tokenizer.save(str(path))
        print(f"Tokenizer saved to '{path}'")
    
    def load(self, path: str | Path) -> None:
        """
        Load a previously saved tokenizer from a JSON file.

        
        Args:
            path (str | Path): Path to the saved tokenizer JSON file.
        
        Raises:
            FileNotFoundError: If the tokenizer file does not exist.
            ValueError: If the file is not a valid tokenizer configuration.
        """
        # Load the tokenizer from the specified file
        self.tokenizer = Tokenizer.from_file(str(path))
        print(f"Tokenizer loaded from '{path}'")
    
    def encode(self, text: str) -> list[int]:
        """
        Encode a text string into a list of token IDs using learned BPE
        
        This method tokenizes the input text by:
        1. Applying learned BPE merge rules to split words into subwords
        2. Converting each subword token to its corresponding vocabulary ID
        3. Returning the sequence of token IDs
        
        Args:
            text (str): The input text to tokenize.
        
        Returns:
            list[int]: A list of token IDs representing the input text.
                    
        Note:
            If the tokenizer hasn't been trained or loaded, this will raise an error.
        """
        # Encode the text using the underlying tokenizer
        # .ids property returns just the token IDs (without attention masks, etc.)
        encoded = self.tokenizer.encode(text).ids
        return encoded

    def decode(self, tokens: list[int]) -> str:
        """
        Decode a list of token IDs back into a human-readable text string.
        
        This method reverses the encoding process by:
        1. Mapping each token ID back to its corresponding subword token
        2. Concatenating the tokens, handling BPE merges appropriately
        3. Returning the reconstructed text string
        
        Args:
            tokens (list[int]): A list of token IDs to decode.
        
        Returns:
            str: The decoded text string.

        Note:
            Special tokens are automatically handled during decoding.
        """
        # Decode the token IDs back to text
        # The tokenizer knows how to properly join BPE subword tokens
        decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return decoded


if __name__ == "__main__":
    VOCAB = 30_000
    
    # Path to the training corpus (should be a plain text file)
    FILEPATH = Path(r'G:\Projects\Python\Onyx\Data\corpus.txt')
    
    # Output path for saving the trained tokenizer
    SAVEPATH = Path(f'BPE_{VOCAB//1000}k.json')
    
    # Initialize the BPE tokenizer with default parameters
    Encoder = BPE()
    
    # Train the tokenizer on the corpus file
    Encoder.train(FILEPATH)
    
    # Save the trained tokenizer for future use
    Encoder.save(SAVEPATH)
    
    # Demonstration: Load the saved tokenizer and test encoding/decoding
    print("\n" + "="*50)
    print("Testing the trained tokenizer:")
    print("="*50)
    
    # Load the tokenizer from the saved file
    Encoder.load(SAVEPATH)
    
    # Test text with contractions and punctuation to verify BPE handling
    text = "Wow; I'm honestly impressed with what you've done so far?!"
    
    # Encode the text to token IDs
    encoded = Encoder.encode(text)
    
    # Decode the token IDs back to text
    decoded = Encoder.decode(encoded)
    
    # Display results for verification
    print(f"Original: {text}")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  {decoded}")
    
    # Verify that encoding/decoding is lossless
    # Note: BPE may normalize spacing/punctuation, so this is informative not assertative
    print(f"\nToken count: {len(encoded)}")
    print(f"Compression ratio: {len(text) / len(encoded):.2f} chars per token")
    
        
        
        
        