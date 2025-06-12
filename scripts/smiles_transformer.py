#!/usr/bin/env python3
"""
SMILES Transformer Embedding Extraction Script

Extract 1024-dimensional molecular embeddings from SMILES strings using custom SMILES Transformer model.

Usage:
    python smiles_transformer.py --input data.csv --smiles_col smiles --output embeddings.npy

Arguments:
    --input: Path to input CSV file
    --smiles_col: Column name for SMILES strings (default: 'smiles')
    --output: Output file path (default: 'smiles_transformer_embeddings.npy')
    --vocab_path: Vocabulary file path (default: 'data/vocab.pkl')
    --model_path: Model weights file path (default: 'data/trfm_12_23000.pkl')
    --batch_size: Batch size (default: 16)
"""

import argparse
import pandas as pd
import numpy as np
import torch
import warnings
import sys
import os
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Add smiles_transformer directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)  # Go up to QSAR directory
smiles_transformer_path = os.path.join(project_dir, 'model', 'smiles_transformer', 'smiles_transformer')
sys.path.append(smiles_transformer_path)

try:
    from pretrain_trfm import TrfmSeq2seq
    from build_vocab import WordVocab
    from utils import split
except ImportError as e:
    print(f"Error: Failed to import required modules: {e}")
    print("Please ensure smiles-transformer directory exists with required module files")
    sys.exit(1)

class SmilesTransformerEmbedder:
    def __init__(self, vocab_path='../model/smiles_transformer/vocab.pkl', model_path='../model/smiles_transformer/trfm_12_23000.pkl'):
        """Initialize SMILES Transformer model"""
        print("Loading SMILES Transformer model...")
        
        # Check if files exist
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load vocabulary
        self.vocab = WordVocab.load_vocab(vocab_path)
        print(f"Vocabulary size: {len(self.vocab)}")
        
        # Initialize model
        self.model = TrfmSeq2seq(len(self.vocab), 256, len(self.vocab), 4)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        # Define special indices
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        
        print("Model loaded successfully!")
    
    def preprocess_smiles(self, smiles_string):
        """
        Preprocess SMILES string
        
        Args:
            smiles_string (str): SMILES string
            
        Returns:
            list: Tokenized list
        """
        # Use split() function to properly tokenize SMILES string
        sm = split(smiles_string)
        tokens = sm.split()
        
        # Sequence length handling - max supported length: 220 tokens
        if len(tokens) > 218:  # Reserve 2 positions for SOS/EOS
            tokens = tokens[:109] + tokens[-109:]
        
        return tokens
    
    def tokens_to_ids(self, tokens):
        """
        Convert tokens to ID sequence
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: ID sequence
        """
        # Convert to ID sequence
        ids = [self.vocab.stoi.get(token, self.unk_index) for token in tokens]
        
        # Add start and end tokens
        ids = [self.sos_index] + ids + [self.eos_index]
        
        # Pad to fixed length 220
        seq_len = 220
        padding = [self.pad_index] * (seq_len - len(ids))
        ids.extend(padding)
        
        return ids
    
    def prepare_batch(self, smiles_list):
        """
        Prepare batch of SMILES strings
        
        Args:
            smiles_list (list): List of SMILES strings
            
        Returns:
            torch.Tensor: Prepared batch tensor
        """
        batch_ids = []
        
        for smiles in smiles_list:
            try:
                tokens = self.preprocess_smiles(smiles)
                ids = self.tokens_to_ids(tokens)
                batch_ids.append(ids)
            except Exception as e:
                print(f"Error processing SMILES '{smiles}': {e}")
                # If error occurs, use padding sequence
                ids = [self.pad_index] * 220
                batch_ids.append(ids)
        
        # Convert to tensor and transpose
        tensor = torch.tensor(batch_ids, dtype=torch.long)  # Shape: (batch_size, seq_len)
        tensor = torch.t(tensor)  # Shape: (seq_len, batch_size) - Transformer required format
        
        return tensor
    
    def get_batch_embeddings(self, smiles_list, batch_size=16):
        """
        Get embeddings for batch of SMILES strings
        
        Args:
            smiles_list (list): List of SMILES strings
            batch_size (int): Batch size
            
        Returns:
            np.ndarray: Embedding matrix (n_molecules, 1024)
        """
        all_embeddings = []
        
        print(f"Processing {len(smiles_list)} molecules, batch size: {batch_size}")
        
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="Extracting embeddings"):
            batch = smiles_list[i:i+batch_size]
            
            # Prepare input tensor
            input_tensor = self.prepare_batch(batch)
            
            # Extract embeddings
            with torch.no_grad():
                try:
                    embeddings = self.model.encode(input_tensor)
                    all_embeddings.append(embeddings)
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    # If error occurs, use zero vectors
                    zero_embeddings = np.zeros((len(batch), 1024))
                    all_embeddings.append(zero_embeddings)
        
        # Combine all batch embeddings
        final_embeddings = np.vstack(all_embeddings)
        
        return final_embeddings

def main():
    parser = argparse.ArgumentParser(description="SMILES Transformer Embedding Extraction Tool")
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV file")
    parser.add_argument("--smiles_col", "-s", default="smiles", help="Column name for SMILES strings (default: smiles)")
    parser.add_argument("--output", "-o", default="smiles_transformer_embeddings.npy", help="Output file path (default: smiles_transformer_embeddings.npy)")
    parser.add_argument("--vocab_path", "-v", default="../model/smiles-transformer/vocab.pkl", help="Vocabulary file path (default: ../model/smiles-transformer/vocab.pkl)")
    parser.add_argument("--model_path", "-m", default="../model/smiles-transformer/trfm_12_23000.pkl", help="Model weights file path (default: ../model/smiles-transformer/trfm_12_23000.pkl)")
    parser.add_argument("--batch_size", "-b", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--save_csv", action="store_true", help="Also save as CSV format")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return
    
    # Read CSV file
    print(f"Reading file: {args.input}")
    try:
        df = pd.read_csv(args.input)
        print(f"Successfully read {len(df)} rows")
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Check if SMILES column exists
    if args.smiles_col not in df.columns:
        print(f"Error: Column '{args.smiles_col}' not found in file")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Get SMILES list
    smiles_list = df[args.smiles_col].dropna().tolist()
    print(f"Found {len(smiles_list)} valid SMILES strings")
    
    # Initialize model
    try:
        embedder = SmilesTransformerEmbedder(args.vocab_path, args.model_path)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    # Extract embeddings
    embeddings = embedder.get_batch_embeddings(smiles_list, args.batch_size)
    
    # Save results
    print(f"Embedding matrix shape: {embeddings.shape}")
    
    # Save as numpy array
    np.save(args.output, embeddings)
    print(f"Embeddings saved to: {args.output}")
    
    # Optional: Save as CSV
    if args.save_csv:
        csv_output = args.output.replace('.npy', '.csv')
        embedding_df = pd.DataFrame(embeddings, columns=[f'dim_{i}' for i in range(embeddings.shape[1])])
        # Add original SMILES
        embedding_df['smiles'] = smiles_list[:len(embeddings)]
        embedding_df.to_csv(csv_output, index=False)
        print(f"Embeddings CSV saved to: {csv_output}")
    
    print("Done!")

if __name__ == "__main__":
    main() 