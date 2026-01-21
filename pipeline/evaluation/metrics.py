"""Metrics calculator for computing evaluation metrics."""

from typing import List, Dict
import networkx as nx

from ..types import Relation, ParsedRelation, EvaluationResult


class MetricsCalculator:
    """Calculates evaluation metrics for relation extraction."""
    
    def calculate_metrics(
        self,
        true_positives: List[Relation],
        false_positives: List[ParsedRelation],
        false_negatives: List[Relation],
        gold_relations: List[Relation],
        predicted_relations: List[ParsedRelation]
    ) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.
        
        Args:
            true_positives: List of correctly predicted relations
            false_positives: List of incorrectly predicted relations
            false_negatives: List of missed gold relations
            gold_relations: All gold relations
            predicted_relations: All predicted relations
            
        Returns:
            Dictionary of metric names to values
        """
        metrics = {}
        
        # Basic metrics
        tp = len(true_positives)
        fp = len(false_positives)
        fn = len(false_negatives)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1_score
        
        # Exact match rate
        exact_match_rate = tp / len(gold_relations) if gold_relations else 0.0
        metrics['exact_match_rate'] = exact_match_rate
        
        # Omission rate (false negatives / total gold)
        omission_rate = fn / len(gold_relations) if gold_relations else 0.0
        metrics['omission_rate'] = omission_rate
        
        # Hallucination rate (false positives / total predicted)
        hallucination_rate = fp / len(predicted_relations) if predicted_relations else 0.0
        metrics['hallucination_rate'] = hallucination_rate
        
        # Redundancy rate (duplicate predictions)
        redundancy_rate = self._calculate_redundancy_rate(predicted_relations)
        metrics['redundancy_rate'] = redundancy_rate
        
        # Graph edit distance
        ged = self._calculate_graph_edit_distance(gold_relations, predicted_relations)
        metrics['graph_edit_distance'] = ged
        
        # Per-type metrics
        per_type_metrics = self._calculate_per_type_metrics(
            true_positives, false_positives, false_negatives
        )
        metrics['per_type_metrics'] = per_type_metrics
        
        return metrics
    
    def _calculate_redundancy_rate(self, predicted_relations: List[ParsedRelation]) -> float:
        """
        Calculate redundancy rate (percentage of duplicate relations).
        
        Args:
            predicted_relations: List of predicted relations
            
        Returns:
            Redundancy rate (0-1)
        """
        if not predicted_relations:
            return 0.0
        
        seen = set()
        duplicates = 0
        
        for rel in predicted_relations:
            if not rel.head_id or not rel.tail_id:
                continue
            
            # Create tuple for comparison
            rel_tuple = (rel.head_id, rel.tail_id, rel.relation_type)
            reverse_tuple = (rel.tail_id, rel.head_id, rel.relation_type)
            
            if rel_tuple in seen or reverse_tuple in seen:
                duplicates += 1
            else:
                seen.add(rel_tuple)
        
        return duplicates / len(predicted_relations) if predicted_relations else 0.0
    
    def _calculate_graph_edit_distance(
        self,
        gold_relations: List[Relation],
        predicted_relations: List[ParsedRelation]
    ) -> float:
        """
        Calculate approximate graph edit distance.
        
        Args:
            gold_relations: List of gold relations
            predicted_relations: List of predicted relations
            
        Returns:
            Graph edit distance (number of edits needed)
        """
        # Build graphs
        gold_graph = self._build_graph(gold_relations)
        pred_graph = self._build_graph_from_parsed(predicted_relations)
        
        # Calculate edit distance as sum of:
        # - Nodes to add/remove
        # - Edges to add/remove
        gold_nodes = set(gold_graph.nodes())
        pred_nodes = set(pred_graph.nodes())
        
        gold_edges = set(gold_graph.edges())
        pred_edges = set(pred_graph.edges())
        
        # Node edits
        nodes_to_add = len(pred_nodes - gold_nodes)
        nodes_to_remove = len(gold_nodes - pred_nodes)
        
        # Edge edits
        edges_to_add = len(pred_edges - gold_edges)
        edges_to_remove = len(gold_edges - pred_edges)
        
        total_edits = nodes_to_add + nodes_to_remove + edges_to_add + edges_to_remove
        
        return float(total_edits)
    
    def _build_graph(self, relations: List[Relation]) -> nx.DiGraph:
        """Build NetworkX graph from relations."""
        graph = nx.DiGraph()
        for rel in relations:
            graph.add_edge(rel.head_id, rel.tail_id, relation_type=rel.type)
        return graph
    
    def _build_graph_from_parsed(self, relations: List[ParsedRelation]) -> nx.DiGraph:
        """Build NetworkX graph from parsed relations."""
        graph = nx.DiGraph()
        for rel in relations:
            if rel.head_id and rel.tail_id:
                graph.add_edge(rel.head_id, rel.tail_id, relation_type=rel.relation_type)
        return graph
    
    def _calculate_per_type_metrics(
        self,
        true_positives: List[Relation],
        false_positives: List[ParsedRelation],
        false_negatives: List[Relation]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics per relation type.
        
        Args:
            true_positives: List of true positives
            false_positives: List of false positives
            false_negatives: List of false negatives
            
        Returns:
            Dictionary mapping relation type to metrics
        """
        # Group by type
        tp_by_type: Dict[str, int] = {}
        fp_by_type: Dict[str, int] = {}
        fn_by_type: Dict[str, int] = {}
        
        for tp in true_positives:
            rel_type = tp.type
            tp_by_type[rel_type] = tp_by_type.get(rel_type, 0) + 1
        
        for fp in false_positives:
            rel_type = fp.relation_type
            fp_by_type[rel_type] = fp_by_type.get(rel_type, 0) + 1
        
        for fn in false_negatives:
            rel_type = fn.type
            fn_by_type[rel_type] = fn_by_type.get(rel_type, 0) + 1
        
        # Calculate metrics per type
        all_types = set(tp_by_type.keys()) | set(fp_by_type.keys()) | set(fn_by_type.keys())
        per_type_metrics = {}
        
        for rel_type in all_types:
            tp = tp_by_type.get(rel_type, 0)
            fp = fp_by_type.get(rel_type, 0)
            fn = fn_by_type.get(rel_type, 0)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_type_metrics[rel_type] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn
            }
        
        return per_type_metrics
