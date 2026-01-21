"""BERTScore-based matcher for semantic relation matching."""

import logging
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..types import Relation, ParsedRelation
from config import Config


class BERTScoreMatcher:
    """Uses BERTScore for semantic similarity-based relation matching."""
    
    def __init__(
        self,
        model_type: str = "microsoft/deberta-xlarge-mnli",
        lang: str = "en",
        use_idf: bool = True,
        batch_size: int = 16,
        logger: Optional[logging.Logger] = None,
        similarity_threshold: float = 0.85,
        use_openai_embeddings: bool = False
    ):
        """
        Initialize BERTScore matcher.
        
        Args:
            model_type: BERT model to use for scoring (or OpenAI embedding model if use_openai_embeddings=True)
            lang: Language code
            use_idf: Whether to use IDF weighting (deprecated - not used in newer bert-score versions)
            batch_size: Batch size for BERTScore computation
            logger: Optional logger instance
            similarity_threshold: Minimum similarity for a match (0-1)
            use_openai_embeddings: If True, use OpenAI embeddings API instead of BERTScore (much faster)
        """
        self.use_openai_embeddings = use_openai_embeddings
        self.model_type = model_type
        self.lang = lang
        self.use_idf = use_idf
        self.batch_size = batch_size
        self.similarity_threshold = similarity_threshold
        self.logger = logger or logging.getLogger(__name__)
        
        if use_openai_embeddings:
            if not OPENAI_AVAILABLE:
                raise ImportError(
                    "OpenAI library is not installed. Install it with: pip install openai"
                )
            if not Config.OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY not found. Required for OpenAI embeddings API."
                )
            self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
            # Use OpenAI embedding model
            self.embedding_model = model_type if model_type.startswith("text-embedding") else Config.RAG_EMBEDDING_MODEL
            # Cache for embeddings to avoid redundant API calls
            self.embedding_cache: Dict[str, List[float]] = {}
            self.logger.info(
                f"[BERTScoreMatcher] Using OpenAI embeddings API (model={self.embedding_model}), "
                f"threshold={similarity_threshold}"
            )
        else:
            if not BERTSCORE_AVAILABLE:
                raise ImportError(
                    "BERTScore is not installed. Install it with: pip install bert-score"
                )
            self.logger.info(
                f"[BERTScoreMatcher] Using BERTScore (model={model_type}), "
                f"threshold={similarity_threshold}"
            )
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))
    
    def _batch_get_embeddings(self, texts: List[str], batch_size: int = 100) -> Dict[str, List[float]]:
        """
        Get embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch (OpenAI limit is ~300k tokens, ~100 texts is safe)
            
        Returns:
            Dictionary mapping text -> embedding vector
        """
        if not self.use_openai_embeddings:
            return {}
        
        # Filter out texts we already have cached
        texts_to_embed = [t for t in texts if t and t not in self.embedding_cache]
        
        if not texts_to_embed:
            return {t: self.embedding_cache[t] for t in texts if t}
        
        self.logger.info(
            f"[BERTScoreMatcher] Getting embeddings for {len(texts_to_embed)} texts "
            f"(batch_size={batch_size})"
        )
        
        all_embeddings = {}
        
        # Process in batches
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                # Store embeddings in cache and return dict
                for j, text in enumerate(batch):
                    embedding = response.data[j].embedding
                    self.embedding_cache[text] = embedding
                    all_embeddings[text] = embedding
            except Exception as e:
                self.logger.warning(
                    f"[BERTScoreMatcher] Error getting embeddings for batch {i//batch_size + 1}: {e}"
                )
                # Fall back to individual calls for this batch
                for text in batch:
                    try:
                        response = self.client.embeddings.create(
                            model=self.embedding_model,
                            input=[text]
                        )
                        embedding = response.data[0].embedding
                        self.embedding_cache[text] = embedding
                        all_embeddings[text] = embedding
                    except Exception as e2:
                        self.logger.warning(
                            f"[BERTScoreMatcher] Error getting embedding for '{text[:50]}...': {e2}"
                        )
        
        # Add cached embeddings to return dict
        for text in texts:
            if text and text in self.embedding_cache:
                all_embeddings[text] = self.embedding_cache[text]
        
        return all_embeddings
    
    def compute_entity_similarity(
        self,
        mention1: str,
        mention2: str,
        embedding1: Optional[List[float]] = None,
        embedding2: Optional[List[float]] = None
    ) -> float:
        """
        Compute similarity between two entity mentions.
        
        Uses OpenAI embeddings API if use_openai_embeddings=True, otherwise BERTScore.
        If embeddings are provided, uses them directly (for batched processing).
        
        Args:
            mention1: First entity mention
            mention2: Second entity mention
            embedding1: Optional pre-computed embedding for mention1
            embedding2: Optional pre-computed embedding for mention2
            
        Returns:
            Similarity score (0-1)
        """
        if not mention1 or not mention2:
            return 0.0
        
        try:
            if self.use_openai_embeddings:
                # Use provided embeddings or get from cache
                if embedding1 is None:
                    embedding1 = self.embedding_cache.get(mention1)
                if embedding2 is None:
                    embedding2 = self.embedding_cache.get(mention2)
                
                # If still not found, get them (shouldn't happen if batching is used)
                if embedding1 is None or embedding2 is None:
                    if embedding1 is None:
                        response = self.client.embeddings.create(
                            model=self.embedding_model,
                            input=[mention1]
                        )
                        embedding1 = response.data[0].embedding
                        self.embedding_cache[mention1] = embedding1
                    if embedding2 is None:
                        response = self.client.embeddings.create(
                            model=self.embedding_model,
                            input=[mention2]
                        )
                        embedding2 = response.data[0].embedding
                        self.embedding_cache[mention2] = embedding2
                
                # Compute cosine similarity
                similarity = self._cosine_similarity(embedding1, embedding2)
                # Normalize to 0-1 range (cosine similarity is already -1 to 1, but embeddings are typically 0-1)
                return max(0.0, similarity)
            else:
                # Use BERTScore
                # BERTScore expects lists of strings
                # Note: use_idf parameter removed as it's not supported in newer versions of bert-score
                P, R, F1 = bert_score(
                    [mention1],
                    [mention2],
                    model_type=self.model_type,
                    lang=self.lang,
                    verbose=False
                )
                # F1 is a tensor, convert to float
                return float(F1.item())
        except Exception as e:
            self.logger.warning(
                f"[BERTScoreMatcher] Error computing similarity between "
                f"'{mention1[:50]}...' and '{mention2[:50]}...': {e}"
            )
            import traceback
            self.logger.debug(traceback.format_exc())
            return 0.0
    
    def compute_relation_similarity(
        self,
        pred_rel: ParsedRelation,
        gold_rel: Relation,
        entity_map = None,
        match_type: bool = True,
        embeddings: Optional[Dict[str, List[float]]] = None
    ) -> float:
        """
        Compute semantic similarity between predicted and gold relations.
        
        Uses BERTScore to compare:
        1. Head entity mentions
        2. Tail entity mentions
        3. Relation types (optional, if match_type=True)
        
        Args:
            pred_rel: Predicted relation
            gold_rel: Gold relation
            entity_map: Optional entity map to get entity mentions for gold relations
            match_type: Whether to include relation type in similarity calculation (default: True)
            embeddings: Optional pre-computed embeddings dict (for batched processing)
            
        Returns:
            Combined similarity score (0-1)
        """
        # Get entity mentions for gold relation
        gold_head_mention = self._get_gold_entity_mention(gold_rel.head_id, entity_map)
        gold_tail_mention = self._get_gold_entity_mention(gold_rel.tail_id, entity_map)
        
        if not gold_head_mention or not gold_tail_mention:
            # Can't compute similarity without mentions
            return 0.0
        
        # Get embeddings if provided
        head_emb1 = embeddings.get(pred_rel.head_mention) if embeddings else None
        head_emb2 = embeddings.get(gold_head_mention) if embeddings else None
        tail_emb1 = embeddings.get(pred_rel.tail_mention) if embeddings else None
        tail_emb2 = embeddings.get(gold_tail_mention) if embeddings else None
        
        # Compute head entity similarity
        head_sim = self.compute_entity_similarity(
            pred_rel.head_mention,
            gold_head_mention,
            embedding1=head_emb1,
            embedding2=head_emb2
        )
        
        # Compute tail entity similarity
        tail_sim = self.compute_entity_similarity(
            pred_rel.tail_mention,
            gold_tail_mention,
            embedding1=tail_emb1,
            embedding2=tail_emb2
        )
        
        if match_type:
            # Check if relation types match (exact match for now, could use semantic similarity)
            type_match = 1.0 if pred_rel.relation_type == gold_rel.type else 0.0
            # Combined score: average of head, tail, and type similarity
            # Weight: 40% head, 40% tail, 20% type
            combined_score = (0.4 * head_sim) + (0.4 * tail_sim) + (0.2 * type_match)
        else:
            # Entity-only matching: 50% head, 50% tail (ignore type)
            combined_score = (0.5 * head_sim) + (0.5 * tail_sim)
        
        return combined_score
    
    def _get_gold_entity_mention(
        self,
        entity_id: str,
        entity_map = None
    ) -> Optional[str]:
        """
        Get entity mention text from entity ID.
        
        Args:
            entity_id: Entity ID
            entity_map: Entity map to look up mentions
            
        Returns:
            Entity mention text or None
        """
        if not entity_map:
            return None
        
        entity = entity_map.get_entity(entity_id)
        if not entity:
            return None
        
        # Prefer canonical name, fall back to first common mention
        if entity.canonical_name:
            return entity.canonical_name
        elif entity.common_mentions:
            return entity.common_mentions[0]
        
        return None
    
    def match_relations_with_bertscore(
        self,
        predicted_relations: List[ParsedRelation],
        gold_relations: List[Relation],
        entity_map = None,
        use_exact_match_first: bool = True,
        match_type: bool = True
    ) -> Tuple[List[Relation], List[ParsedRelation], List[Relation], List[Tuple[ParsedRelation, Relation, float]]]:
        """
        Match relations using BERTScore for semantic similarity.
        
        Args:
            predicted_relations: List of predicted relations
            gold_relations: List of gold relations
            entity_map: Optional entity map for getting gold entity mentions
            use_exact_match_first: If True, try exact ID matching first, then BERTScore
            match_type: Whether to require relation type to match (default: True)
            
        Returns:
            Tuple of (true_positives, false_positives, false_negatives, semantic_matches)
            where semantic_matches are (predicted, gold, similarity_score) tuples
        """
        matched_gold_indices = set()
        true_positives = []
        false_positives = []
        semantic_matches = []  # Relations matched by BERTScore but not exact match
        
        # First pass: exact ID matching (if enabled)
        if use_exact_match_first:
            for i, pred_rel in enumerate(predicted_relations):
                if not pred_rel.head_id or not pred_rel.tail_id:
                    continue
                
                for j, gold_rel in enumerate(gold_relations):
                    if j in matched_gold_indices:
                        continue
                    
                    # Check exact match
                    if (pred_rel.head_id == gold_rel.head_id and 
                        pred_rel.tail_id == gold_rel.tail_id and
                        pred_rel.relation_type == gold_rel.type):
                        true_positives.append(gold_rel)
                        matched_gold_indices.add(j)
                        break
        
        # Second pass: BERTScore matching for unmatched relations
        total_comparisons = sum(
            len([g for j, g in enumerate(gold_relations) if j not in matched_gold_indices])
            for pred_rel in predicted_relations
            if not any(
                pred_rel.head_id == tp.head_id and 
                pred_rel.tail_id == tp.tail_id and
                pred_rel.relation_type == tp.type
                for tp in true_positives
            )
        )
        comparison_count = 0
        
        self.logger.info(
            f"[BERTScoreMatcher] Starting similarity matching: "
            f"{len(predicted_relations)} predicted relations, "
            f"{len(gold_relations)} gold relations, "
            f"~{total_comparisons} comparisons needed"
        )
        
        # Pre-compute embeddings for all entity mentions if using OpenAI embeddings
        embeddings: Dict[str, List[float]] = {}
        if self.use_openai_embeddings:
            # Collect all unique entity mentions
            all_mentions: Set[str] = set()
            for pred_rel in predicted_relations:
                if pred_rel.head_mention:
                    all_mentions.add(pred_rel.head_mention)
                if pred_rel.tail_mention:
                    all_mentions.add(pred_rel.tail_mention)
            
            for gold_rel in gold_relations:
                gold_head = self._get_gold_entity_mention(gold_rel.head_id, entity_map)
                gold_tail = self._get_gold_entity_mention(gold_rel.tail_id, entity_map)
                if gold_head:
                    all_mentions.add(gold_head)
                if gold_tail:
                    all_mentions.add(gold_tail)
            
            # Batch get all embeddings
            self.logger.info(
                f"[BERTScoreMatcher] Pre-computing embeddings for {len(all_mentions)} unique entity mentions"
            )
            embeddings = self._batch_get_embeddings(list(all_mentions))
            self.logger.info(
                f"[BERTScoreMatcher] Got {len(embeddings)} embeddings (cached: {len(self.embedding_cache)})"
            )
        
        for i, pred_rel in enumerate(predicted_relations):
            # Skip if already matched exactly
            if any(
                pred_rel.head_id == tp.head_id and 
                pred_rel.tail_id == tp.tail_id and
                pred_rel.relation_type == tp.type
                for tp in true_positives
            ):
                continue
            
            best_match = None
            best_score = 0.0
            best_gold_idx = None
            
            for j, gold_rel in enumerate(gold_relations):
                if j in matched_gold_indices:
                    continue
                
                comparison_count += 1
                if comparison_count % 10 == 0:
                    self.logger.info(
                        f"[BERTScoreMatcher] Progress: {comparison_count}/{total_comparisons} "
                        f"comparisons ({100*comparison_count/total_comparisons:.1f}%)"
                    )
                
                # Compute BERTScore similarity (using pre-computed embeddings if available)
                similarity = self.compute_relation_similarity(
                    pred_rel,
                    gold_rel,
                    entity_map,
                    match_type=match_type,
                    embeddings=embeddings if self.use_openai_embeddings else None
                )
                
                if similarity > best_score and similarity >= self.similarity_threshold:
                    best_score = similarity
                    best_match = gold_rel
                    best_gold_idx = j
            
            if best_match:
                # Check if it's an exact entity match
                is_exact = (
                    pred_rel.head_id == best_match.head_id and
                    pred_rel.tail_id == best_match.tail_id
                )
                
                if is_exact:
                    # Exact entity match, just type might differ
                    true_positives.append(best_match)
                    matched_gold_indices.add(best_gold_idx)
                else:
                    # Semantic match (entities are similar but not exact)
                    # Count as true positive if similarity is high enough (above threshold)
                    # This allows BERTScore to help when entity resolution fails
                    true_positives.append(best_match)
                    matched_gold_indices.add(best_gold_idx)
                    # Also track as semantic match for analysis
                    semantic_matches.append((pred_rel, best_match, best_score))
            else:
                # No match found
                false_positives.append(pred_rel)
        
        # False negatives: unmatched gold relations
        false_negatives = [
            gold_rel for i, gold_rel in enumerate(gold_relations)
            if i not in matched_gold_indices
        ]
        
        return true_positives, false_positives, false_negatives, semantic_matches
    
    def compute_bertscore_metric(
        self,
        predicted_relations: List[ParsedRelation],
        gold_relations: List[Relation],
        entity_map = None
    ) -> float:
        """
        Compute average BERTScore for all predicted relations against their best gold matches.
        
        Args:
            predicted_relations: List of predicted relations
            gold_relations: List of gold relations
            entity_map: Optional entity map
            
        Returns:
            Average BERTScore F1 across all relations
        """
        if not predicted_relations or not gold_relations:
            return 0.0
        
        scores = []
        matched_gold_indices = set()
        
        for pred_rel in predicted_relations:
            best_score = 0.0
            best_idx = None
            
            for i, gold_rel in enumerate(gold_relations):
                if i in matched_gold_indices:
                    continue
                
                score = self.compute_relation_similarity(
                    pred_rel,
                    gold_rel,
                    entity_map,
                    match_type=True  # For metric computation, always include type
                )
                
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            if best_score > 0:
                scores.append(best_score)
                if best_idx is not None:
                    matched_gold_indices.add(best_idx)
        
        return np.mean(scores) if scores else 0.0

