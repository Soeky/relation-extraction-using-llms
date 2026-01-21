"""Global entity map for aggregating entities across documents."""

import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter

from ..types import GlobalEntity, Entity, Mention, GoldRelations


class GlobalEntityMap:
    """Maps global entity IDs to aggregated entity information."""
    
    def __init__(self):
        """Initialize empty entity map."""
        self.entities: Dict[str, GlobalEntity] = {}
    
    def build_from_gold_relations(self, gold_relations_list: List[GoldRelations]) -> None:
        """
        Build entity map from gold relations.
        
        Args:
            gold_relations_list: List of GoldRelations objects
        """
        self.entities = {}
        
        for gold_rel in gold_relations_list:
            for entity in gold_rel.entities:
                entity_id = entity.id
                
                if entity_id not in self.entities:
                    # Create new global entity
                    global_entity = GlobalEntity(
                        id=entity_id,
                        type=entity.type,
                        all_mentions=[],
                        common_mentions=[],
                        document_count=0,
                        canonical_name=""
                    )
                    self.entities[entity_id] = global_entity
                
                # Add mentions
                global_entity = self.entities[entity_id]
                global_entity.all_mentions.extend(entity.mentions)
                
                # Track document count (increment only once per document)
                # We'll recalculate this at the end
        
        # Post-process: calculate common mentions and document counts
        self._post_process()
    
    def _post_process(self) -> None:
        """Post-process entities to calculate statistics."""
        # Track which documents contain each entity
        entity_docs: Dict[str, set] = {}
        
        for entity_id, global_entity in self.entities.items():
            # Count mention texts
            mention_texts = [m.text for m in global_entity.all_mentions]
            mention_counter = Counter(mention_texts)
            
            # Get most common mentions
            common_mentions = [text for text, count in mention_counter.most_common(10)]
            global_entity.common_mentions = common_mentions
            
            # Set canonical name (most common mention)
            if common_mentions:
                global_entity.canonical_name = common_mentions[0]
            
            # Count documents (approximate by counting unique passage_index combinations)
            # For simplicity, we'll use a heuristic based on mention diversity
            unique_mentions = len(set(mention_texts))
            global_entity.document_count = max(1, unique_mentions // 2)  # Rough estimate
    
    def get_entity(self, entity_id: str) -> Optional[GlobalEntity]:
        """
        Get entity by ID.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            GlobalEntity or None if not found
        """
        return self.entities.get(entity_id)
    
    def find_entity_by_mention(
        self, 
        mention_text: str, 
        entity_type: Optional[str] = None,
        fuzzy: bool = True
    ) -> List[GlobalEntity]:
        """
        Find entities by mention text.
        
        Args:
            mention_text: Text to search for
            entity_type: Optional entity type filter
            fuzzy: Whether to use fuzzy matching (case-insensitive, partial)
            
        Returns:
            List of matching GlobalEntity objects
        """
        mention_lower = mention_text.lower().strip()
        matches = []
        
        for entity_id, global_entity in self.entities.items():
            # Type filter
            if entity_type and global_entity.type != entity_type:
                continue
            
            # Check mentions
            for mention in global_entity.all_mentions:
                mention_text_lower = mention.text.lower().strip()
                
                if fuzzy:
                    # Case-insensitive partial match
                    if mention_lower in mention_text_lower or mention_text_lower in mention_lower:
                        matches.append(global_entity)
                        break
                    # Also check common mentions
                    if any(mention_lower in cm.lower() or cm.lower() in mention_lower 
                           for cm in global_entity.common_mentions):
                        matches.append(global_entity)
                        break
                else:
                    # Exact match
                    if mention_text_lower == mention_lower:
                        matches.append(global_entity)
                        break
        
        return matches
    
    def __len__(self) -> int:
        """Return number of entities in map."""
        return len(self.entities)
    
    def __iter__(self):
        """Iterate over entities."""
        return iter(self.entities.values())
    
    def save(self, file_path: Path) -> None:
        """
        Save entity map to JSON file.
        
        Args:
            file_path: Path to save file
        """
        data = {
            "entities": []
        }
        
        for entity_id, global_entity in self.entities.items():
            entity_data = {
                "id": global_entity.id,
                "type": global_entity.type,
                "canonical_name": global_entity.canonical_name,
                "document_count": global_entity.document_count,
                "common_mentions": global_entity.common_mentions,
                "mention_count": len(global_entity.all_mentions)
            }
            data["entities"].append(entity_data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, file_path: Path) -> "GlobalEntityMap":
        """
        Load entity map from JSON file.
        
        Args:
            file_path: Path to load file
            
        Returns:
            GlobalEntityMap instance
        """
        entity_map = cls()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for entity_data in data.get("entities", []):
            global_entity = GlobalEntity(
                id=entity_data["id"],
                type=entity_data["type"],
                all_mentions=[],  # Not stored in summary
                common_mentions=entity_data.get("common_mentions", []),
                document_count=entity_data.get("document_count", 0),
                canonical_name=entity_data.get("canonical_name", "")
            )
            entity_map.entities[entity_data["id"]] = global_entity
        
        return entity_map
