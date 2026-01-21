"""Data loading components."""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional

from ..types import Document, GoldRelations, Entity, Relation, Mention


class DocumentLoader:
    """Loads documents from text files."""
    
    def __init__(self, clean_text_path: Path):
        """
        Initialize document loader.
        
        Args:
            clean_text_path: Path to clean_text directory
        """
        self.clean_text_path = clean_text_path
    
    def load(self, split: str) -> List[Document]:
        """
        Load documents from a split.
        
        Args:
            split: Split name ("dev", "test", or "train")
            
        Returns:
            List of Document objects
        """
        # Determine directory name based on split
        if split in ("dev", "test", "train"):
            dir_name = split
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'dev', 'test', or 'train'")
        
        split_dir = self.clean_text_path / dir_name
        if not split_dir.exists():
            raise FileNotFoundError(f"Directory not found: {split_dir}")
        
        documents = []
        for txt_file in sorted(split_dir.glob("*.txt")):
            doc_id = txt_file.stem
            
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # First line is title, rest is body
            if len(lines) >= 1:
                title = lines[0].strip()
                body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
                text = f"{title}\n\n{body}" if body else title
            else:
                title = ""
                body = ""
                text = ""
            
            doc = Document(
                doc_id=doc_id,
                text=text,
                title=title,
                body=body
            )
            documents.append(doc)
        
        return documents


class GoldRelationsLoader:
    """Loads gold standard relations from JSON files."""
    
    def __init__(self, gold_relations_path: Path, logger: Optional[logging.Logger] = None):
        """
        Initialize gold relations loader.
        
        Args:
            gold_relations_path: Path to gold_relations directory
            logger: Optional logger instance
        """
        self.gold_relations_path = gold_relations_path
        self.logger = logger or logging.getLogger(__name__)
    
    def load(self, split: str) -> List[GoldRelations]:
        """
        Load gold relations from a split.
        
        Args:
            split: Split name ("dev", "test", or "train")
            
        Returns:
            List of GoldRelations objects
        """
        split_dir = self.gold_relations_path / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Directory not found: {split_dir}")
        
        gold_relations_list = []
        for json_file in sorted(split_dir.glob("*.json")):
            self.logger.debug(f"[GoldRelationsLoader] Loading gold relations from: {json_file.name}")
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            doc_id = data.get("doc_id", json_file.stem)
            title = data.get("title", "")
            body = data.get("body", "")
            
            # Parse entities
            entities = []
            for ent_data in data.get("entities", []):
                entity_id = ent_data.get("id", "")
                entity_type = ent_data.get("type", "")
                
                mentions = []
                for mention_data in ent_data.get("mentions", []):
                    mention = Mention(
                        text=mention_data.get("text", ""),
                        passage_index=mention_data.get("passage_index", 0),
                        passage_offset=mention_data.get("passage_offset", 0),
                        char_offset=mention_data.get("char_offset", 0),
                        length=mention_data.get("length", 0)
                    )
                    mentions.append(mention)
                
                entity = Entity(
                    id=entity_id,
                    type=entity_type,
                    mentions=mentions
                )
                entities.append(entity)
            
            # Parse relations
            relations = []
            for rel_data in data.get("relations", []):
                relation = Relation(
                    id=rel_data.get("id", ""),
                    head_id=rel_data.get("head_id", ""),
                    tail_id=rel_data.get("tail_id", ""),
                    type=rel_data.get("type", ""),
                    novel=rel_data.get("novel", "No")
                )
                relations.append(relation)
            
            gold_relations = GoldRelations(
                doc_id=doc_id,
                entities=entities,
                relations=relations,
                title=title,
                body=body,
                file_path=str(json_file)
            )
            gold_relations_list.append(gold_relations)
            
            self.logger.debug(
                f"[GoldRelationsLoader] Loaded {doc_id}: "
                f"{len(entities)} entities, {len(relations)} relations from {json_file.name}"
            )
        
        self.logger.info(f"[GoldRelationsLoader] Loaded {len(gold_relations_list)} gold relation files")
        return gold_relations_list


class DatasetLoader:
    """Combines document and gold relations loading."""
    
    def __init__(
        self, 
        clean_text_path: Path, 
        gold_relations_path: Path,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize dataset loader.
        
        Args:
            clean_text_path: Path to clean_text directory
            gold_relations_path: Path to gold_relations directory
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.document_loader = DocumentLoader(clean_text_path)
        self.gold_relations_loader = GoldRelationsLoader(gold_relations_path, logger=self.logger)
    
    def load(self, split: str) -> Tuple[List[Document], List[GoldRelations]]:
        """
        Load documents and gold relations for a split.
        
        Args:
            split: Split name ("dev", "test", or "train")
            
        Returns:
            Tuple of (documents, gold_relations) lists
        """
        documents = self.document_loader.load(split)
        gold_relations = self.gold_relations_loader.load(split)
        
        # Create a mapping of doc_id to gold_relations for matching
        gold_map = {gr.doc_id: gr for gr in gold_relations}
        
        # Match documents to gold relations and ensure order
        matched_documents = []
        matched_gold_relations = []
        
        for doc in documents:
            if doc.doc_id in gold_map:
                matched_documents.append(doc)
                matched_gold_relations.append(gold_map[doc.doc_id])
            else:
                # Document without gold relations - still include it
                matched_documents.append(doc)
                # Create empty gold relations
                empty_gold = GoldRelations(
                    doc_id=doc.doc_id,
                    entities=[],
                    relations=[],
                    title=doc.title,
                    body=doc.body
                )
                matched_gold_relations.append(empty_gold)
        
        return matched_documents, matched_gold_relations
