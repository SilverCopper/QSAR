#!/usr/bin/env python3
"""
ChemBERTa Embedding Extraction Script

Extract 768-dimensional molecular embeddings from SMILES strings using pretrained ChemBERTa model.

Usage:
    python chembert.py --input data.csv --smiles_col smiles --output embeddings.npy

Arguments:
    --input: Path to input CSV file
    --smiles_col: Column name for SMILES strings (default: 'smiles')
    --output: Output file path (default: 'chembert_embeddings.npy')
    --batch_size: Batch size (default: 32)
"""

import argparse
import pandas as pd
import numpy as np
import torch
import warnings
from transformers import RobertaModel, RobertaTokenizer
from tqdm import tqdm
import os

warnings.filterwarnings("ignore")

class ChemBERTaEmbedder:
    def __init__(self):
        """Initialize ChemBERTa model and tokenizer"""
        print("Loading ChemBERTa model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.model_name = "seyonec/PubChem10M_SMILES_BPE_450k"
        self.model = RobertaModel.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        
        print("Model loaded successfully!")
    
    def get_embedding(self, smiles):
        """
        Get embedding for a single SMILES string
        
        Args:
            smiles (str): SMILES string
            
        Returns:
            np.ndarray: 768-dimensional embedding vector
        """
        # Tokenize SMILES string
        inputs = self.tokenizer(smiles, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get last hidden states
        last_hidden_states = outputs.last_hidden_state
        
        # Average all token embeddings to get molecule embedding
        embedding_avg = torch.mean(last_hidden_states[0], dim=0)
        
        return embedding_avg.cpu().numpy()
    
    def get_batch_embeddings(self, smiles_list, batch_size=32):
        """
        Get embeddings for batch of SMILES strings
        
        Args:
            smiles_list (list): List of SMILES strings
            batch_size (int): Batch size
            
        Returns:
            np.ndarray: Embedding matrix (n_molecules, 768)
        """
        embeddings = []
        
        print(f"Processing {len(smiles_list)} molecules, batch size: {batch_size}")
        
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="Extracting embeddings"):
            batch = smiles_list[i:i+batch_size]
            batch_embeddings = []
            
            for smiles in batch:
                try:
                    embedding = self.get_embedding(smiles)
                    batch_embeddings.append(embedding)
                except Exception as e:
                    print(f"Error processing SMILES '{smiles}': {e}")
                    # Use zero vector if error occurs
                    batch_embeddings.append(np.zeros(768))
            
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)

def main():
    parser = argparse.ArgumentParser(description="ChemBERTa Embedding Extraction Tool")
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV file")
    parser.add_argument("--smiles_col", "-s", default="smiles", help="Column name for SMILES strings (default: smiles)")
    parser.add_argument("--output", "-o", default="chembert_embeddings.npy", help="Output file path (default: chembert_embeddings.npy)")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size (default: 32)")
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
    embedder = ChemBERTaEmbedder()
    
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