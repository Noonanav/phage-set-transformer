"""
Data loading and processing utilities for the phage-set-transformer package.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


def load_embeddings_flexible(embedding_dir: str, 
                           genome_ids_csv: Optional[str] = None, 
                           id_column: str = "genome_id") -> Dict[str, Tuple[np.ndarray, List[str]]]:
    """
    Load embeddings from a directory with support for multiple formats.
    
    Args:
        embedding_dir: Path to directory containing .npy files
        genome_ids_csv: Optional CSV file with specific genome IDs to load
        id_column: Column name in CSV containing genome IDs
        
    Returns:
        Dictionary mapping genome IDs to (embedding_array, gene_ids) tuples
    """
    embedding_dir = Path(embedding_dir)
    embeddings = {}

    # If genome_ids_csv is provided, load specific genome IDs
    genome_ids = None
    if genome_ids_csv:
        try:
            genome_df = pd.read_csv(genome_ids_csv)
            if id_column not in genome_df.columns:
                raise ValueError(f"Column '{id_column}' not found in {genome_ids_csv}")
            genome_ids = set(genome_df[id_column].astype(str))
            logger.info(f"Loaded {len(genome_ids)} genome IDs from {genome_ids_csv}")
        except Exception as e:
            logger.error(f"Error loading genome IDs: {e}")
            logger.info("Falling back to loading all embeddings")
            genome_ids = None

    # Get list of all embedding files
    file_paths = list(embedding_dir.glob('*.npy'))
    if not file_paths:
        logger.warning(f"No .npy files found in {embedding_dir}")
        return embeddings

    # Track which genomes were found
    found_genomes = set()

    # Process each file
    for file_path in file_paths:
        identifier = file_path.stem  # filename without extension

        # Skip if not in the list of genome IDs (if a list was provided)
        if genome_ids is not None and identifier not in genome_ids:
            continue

        try:
            embedding_data = np.load(file_path, allow_pickle=True)

            # Detect and process based on file format
            if isinstance(embedding_data, np.ndarray):
                # If it's a simple array with shape [num_genes, embedding_dim]
                if embedding_data.ndim == 2:
                    # Create generic gene IDs
                    gene_ids = [f"{identifier}_gene_{i+1}" for i in range(embedding_data.shape[0])]
                    embeddings[identifier] = (embedding_data, gene_ids)
                    format_type = "simple_array"

                # If it's a dictionary-like object in a 0-d array
                elif hasattr(embedding_data.item(), 'keys'):
                    embeddings_dict = embedding_data.item()
                    gene_ids = list(embeddings_dict.keys())
                    embedding_list = [embeddings_dict[gene_id] for gene_id in gene_ids]

                    if not embedding_list:
                        logger.warning(f"No embeddings found in {file_path}")
                        continue

                    embeddings_array = np.stack(embedding_list)
                    embeddings[identifier] = (embeddings_array, gene_ids)
                    format_type = "dict_format"
                else:
                    logger.warning(f"Unrecognized array format for {file_path}")
                    continue
            else:
                logger.warning(f"Unrecognized data type for {file_path}")
                continue

            found_genomes.add(identifier)
            logger.debug(f"Loaded {identifier} ({format_type}): {embeddings[identifier][0].shape}")

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")

    # Report on results
    logger.info(f"Successfully loaded {len(embeddings)} embeddings")

    # If genome IDs were provided, report on missing genomes
    if genome_ids is not None:
        missing_genomes = genome_ids - found_genomes
        if missing_genomes:
            logger.warning(f"{len(missing_genomes)} genomes from the CSV were not found")

    return embeddings


def filter_interactions_by_strain(interactions_df: pd.DataFrame, 
                                random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split interactions into train/test by strain.
    
    Args:
        interactions_df: DataFrame with columns 'strain', 'phage', 'interaction'
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    unique_strains = interactions_df['strain'].unique()
    train_strains, test_strains = train_test_split(
        unique_strains,
        test_size=0.2,
        random_state=random_state
    )

    train_df = interactions_df[interactions_df['strain'].isin(train_strains)]
    test_df = interactions_df[interactions_df['strain'].isin(test_strains)]

    logger.info(f"Train set: {len(train_df)} interactions, {len(train_strains)} strains")
    logger.info(f"Test set:  {len(test_df)} interactions, {len(test_strains)} strains")
    
    return train_df, test_df


def calculate_phage_specific_weights(train_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate positive weights for each phage based on its interaction distribution.
    
    Args:
        train_df: Training DataFrame with phage interactions
        
    Returns:
        Dictionary mapping phage IDs to weights
    """
    phage_weights = {}

    for phage in train_df['phage'].unique():
        phage_df = train_df[train_df['phage'] == phage]
        num_pos = (phage_df['interaction'] == 1).sum()
        num_neg = (phage_df['interaction'] == 0).sum()

        if num_pos > 0:
            # Calculate the weight as negative/positive ratio
            weight = num_neg / num_pos
            # Clip extremely high values
            weight = min(max(weight, 1.0), 10.0)
        else:
            # If no positive examples, use a default high weight
            weight = 5.0

        phage_weights[phage] = weight

    logger.info(f"Calculated weights for {len(phage_weights)} phages")
    return phage_weights


class StrainPhageDatasetWithWeights(Dataset):
    """Dataset for strain-phage interactions with phage-specific weights."""
    
    def __init__(self, 
                 interactions_df: pd.DataFrame, 
                 strain_embeddings: Dict[str, Tuple[np.ndarray, List[str]]], 
                 phage_embeddings: Dict[str, Tuple[np.ndarray, List[str]]], 
                 phage_weights: Dict[str, float]):
        """
        Initialize the dataset.
        
        Args:
            interactions_df: DataFrame with interaction data
            strain_embeddings: Dictionary of strain embeddings
            phage_embeddings: Dictionary of phage embeddings  
            phage_weights: Dictionary of phage-specific weights
        """
        self.interactions = interactions_df.reset_index(drop=True)
        self.strain_embeddings = strain_embeddings
        self.phage_embeddings = phage_embeddings
        self.phage_weights = phage_weights

        # Check for missing embeddings
        missing_strains = set(self.interactions['strain']) - set(strain_embeddings.keys())
        missing_phages = set(self.interactions['phage']) - set(phage_embeddings.keys())

        if missing_strains or missing_phages:
            raise ValueError(
                f"Missing embeddings for {len(missing_strains)} strains "
                f"and {len(missing_phages)} phages."
            )

    def __len__(self) -> int:
        return len(self.interactions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
        row = self.interactions.iloc[idx]
        strain_emb = torch.tensor(
            self.strain_embeddings[row['strain']][0], dtype=torch.float32
        )  # shape [n_s, embedding_dim]
        phage_emb = torch.tensor(
            self.phage_embeddings[row['phage']][0], dtype=torch.float32
        )   # shape [n_p, embedding_dim]
        label = torch.tensor(row['interaction'], dtype=torch.float32)
        weight = torch.tensor(self.phage_weights.get(row['phage'], 1.0), dtype=torch.float32)

        return strain_emb, phage_emb, label, weight, row['strain'], row['phage']


def collate_variable_sets_with_weights(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    """
    Collate function for variable-length sets with weights, optimized for large sets.
    
    Args:
        batch: List of tuples from dataset
        
    Returns:
        Tuple of batched tensors
    """
    # Unpack the batch
    strains, phages, labels, weights, strain_ids, phage_ids = zip(*batch)

    # Find max set sizes
    max_strain_len = max(s.shape[0] for s in strains)
    max_phage_len = max(p.shape[0] for p in phages)

    # Emb dim (assume consistent)
    emb_dim = strains[0].shape[1]
    batch_size = len(batch)

    # Allocate zero-padded Tensors + boolean masks
    strain_padded = torch.zeros(batch_size, max_strain_len, emb_dim, dtype=torch.float32)
    phage_padded = torch.zeros(batch_size, max_phage_len, emb_dim, dtype=torch.float32)

    strain_mask = torch.zeros(batch_size, max_strain_len, dtype=torch.bool)
    phage_mask = torch.zeros(batch_size, max_phage_len, dtype=torch.bool)

    label_batch = torch.zeros(batch_size, 1, dtype=torch.float32)
    weight_batch = torch.zeros(batch_size, 1, dtype=torch.float32)

    # Copy each sample's data into padded Tensors
    for i, (s_emb, p_emb, label, weight) in enumerate(zip(strains, phages, labels, weights)):
        s_len = s_emb.shape[0]
        p_len = p_emb.shape[0]

        strain_padded[i, :s_len, :] = s_emb
        phage_padded[i, :p_len, :] = p_emb
        strain_mask[i, :s_len] = True
        phage_mask[i, :p_len] = True

        label_batch[i, 0] = float(label)
        weight_batch[i, 0] = float(weight)

    return strain_padded, phage_padded, strain_mask, phage_mask, label_batch, weight_batch, strain_ids, phage_ids

def stratify_dataframe(df, batch_size, random_state=None):
    """Reorder DataFrame so positives are distributed evenly across batches."""
    
    # Separate positive and negative examples with controlled shuffling
    pos_df = df[df['interaction'] == 1].sample(frac=1, random_state=random_state).reset_index(drop=True)
    neg_df = df[df['interaction'] == 0].sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Handle edge case of no positives
    if len(pos_df) == 0:
        return neg_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Calculate batch distribution parameters
    total_samples = len(df)
    n_batches = (total_samples + batch_size - 1) // batch_size
    pos_per_batch = len(pos_df) // n_batches
    extra_pos = len(pos_df) % n_batches
    
    # Initialize result collection and tracking variables
    result_rows = []
    pos_idx = neg_idx = 0
    rng = np.random.RandomState(random_state) if random_state is not None else np.random
    
    # Process each batch sequentially
    for batch_num in range(n_batches):
        actual_batch_size = min(batch_size, total_samples - batch_num * batch_size)
        num_pos = pos_per_batch + (1 if batch_num < extra_pos else 0)
        num_pos = min(num_pos, actual_batch_size)
        
        batch_rows = []
        
        # Add positive examples
        for _ in range(num_pos):
            if pos_idx < len(pos_df):
                batch_rows.append(pos_df.iloc[pos_idx].to_dict())
                pos_idx += 1
        
        # Add negative examples
        for _ in range(actual_batch_size - num_pos):
            if neg_idx < len(neg_df):
                batch_rows.append(neg_df.iloc[neg_idx].to_dict())
                neg_idx += 1
        
        # Shuffle within batch and add to results
        rng.shuffle(batch_rows)
        result_rows.extend(batch_rows)
    
    return pd.DataFrame(result_rows)

def create_data_loaders(train_df: Optional[pd.DataFrame], 
                       test_df: pd.DataFrame, 
                       strain_embeddings: Dict[str, Tuple[np.ndarray, List[str]]], 
                       phage_embeddings: Dict[str, Tuple[np.ndarray, List[str]]],
                       batch_size: int = 16, 
                       use_phage_weights: bool = True,
                       random_state: Optional[int] = None) -> Tuple[Optional[DataLoader], DataLoader]:
    """
    Create data loaders optimized for single-threaded environment.
    
    Args:
        train_df: Training DataFrame (can be None)
        test_df: Test DataFrame
        strain_embeddings: Dictionary of strain embeddings
        phage_embeddings: Dictionary of phage embeddings
        batch_size: Batch size for data loaders
        use_phage_weights: Whether to use phage-specific weights
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_loader = None
    phage_weights = {}

    # Explicitly set num_workers to 0 for compatibility
    num_workers = 0

    if train_df is not None and not train_df.empty:
        # Apply stratified sampling by reordering DataFrame
        train_df = stratify_dataframe(train_df, batch_size, random_state)

        # Calculate phage-specific weights from training data if enabled
        if use_phage_weights:
            phage_weights = calculate_phage_specific_weights(train_df)
        else:
            # Use a default weight of 1.0 for all phages
            for phage in train_df['phage'].unique():
                phage_weights[phage] = 1.0
            logger.info("Using default weight of 1.0 for all phages")

        # Create dataset with weights
        train_dataset = StrainPhageDatasetWithWeights(
            train_df, strain_embeddings, phage_embeddings, phage_weights
        )

        # Create data loader with custom collate function, no multiprocessing
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,  # Still useful for GPU data transfer
            collate_fn=collate_variable_sets_with_weights
        )

    # For test data, use the same weights calculated from training data
    test_dataset = StrainPhageDatasetWithWeights(
        test_df, strain_embeddings, phage_embeddings, phage_weights
    )

    # Create test data loader, no multiprocessing
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_variable_sets_with_weights
    )

    return train_loader, test_loader
