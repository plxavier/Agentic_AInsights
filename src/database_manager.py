# database_manager.py - Manages switching between legacy and structured databases
import os
import sys
from typing import Optional, Union, List, Dict, Any
from langchain.schema import Document

# Import both databases
try:
    from database import VectorDatabase as LegacyDatabase
    from database import DocumentLoader as LegacyDocumentLoader

    LEGACY_AVAILABLE = True
except ImportError:
    print("⚠️ Legacy database.py not found")
    LEGACY_AVAILABLE = False

try:
    from database_structured import StructuredVectorDatabase, StructuredDocumentLoader

    STRUCTURED_AVAILABLE = True
except ImportError:
    print("⚠️ Structured database not found")
    STRUCTURED_AVAILABLE = False


class DatabaseManager:
    """Manages switching between legacy and structured databases"""

    def __init__(self, mode: str = "auto"):
        """
        Initialize database manager

        Args:
            mode: "legacy", "structured", or "auto"
        """
        self.mode = mode
        self.current_db = None
        self.current_db_type = None

        # Initialize based on mode
        if mode == "legacy" and LEGACY_AVAILABLE:
            self.current_db = LegacyDatabase()
            self.current_db_type = "legacy"
            print("📚 Using LEGACY database")

        elif mode == "structured" and STRUCTURED_AVAILABLE:
            self.current_db = StructuredVectorDatabase()
            self.current_db_type = "structured"
            print("📚 Using STRUCTURED database")

        elif mode == "auto":
            # Auto-select: prefer structured if available
            if STRUCTURED_AVAILABLE:
                self.current_db = StructuredVectorDatabase()
                self.current_db_type = "structured"
                print("📚 Auto-selected STRUCTURED database")
            elif LEGACY_AVAILABLE:
                self.current_db = LegacyDatabase()
                self.current_db_type = "legacy"
                print("📚 Auto-selected LEGACY database")
            else:
                raise ImportError("No database modules available")
        else:
            available_modes = []
            if LEGACY_AVAILABLE:
                available_modes.append("legacy")
            if STRUCTURED_AVAILABLE:
                available_modes.append("structured")
            raise ValueError(f"Mode '{mode}' not available. Available: {available_modes}")

    def switch_mode(self, new_mode: str):
        """Switch between legacy and structured modes"""
        if new_mode == self.mode:
            return

        old_db = self.current_db
        old_type = self.current_db_type

        try:
            if new_mode == "legacy" and LEGACY_AVAILABLE:
                self.current_db = LegacyDatabase()
                self.current_db_type = "legacy"
                self.mode = "legacy"
                print(f"🔄 Switched from {old_type} to LEGACY database")

            elif new_mode == "structured" and STRUCTURED_AVAILABLE:
                self.current_db = StructuredVectorDatabase()
                self.current_db_type = "structured"
                self.mode = "structured"
                print(f"🔄 Switched from {old_type} to STRUCTURED database")

            else:
                print(f"⚠️ Cannot switch to {new_mode}")
                return False

            return True

        except Exception as e:
            print(f"❌ Error switching to {new_mode}: {e}")
            # Revert to old
            self.current_db = old_db
            self.current_db_type = old_type
            return False

    def add_paper(self, text: str, metadata: Dict = None, **kwargs) -> Dict:
        """Add paper using current database"""
        if self.current_db_type == "legacy":
            return self.current_db.safe_add_paper(text, metadata, **kwargs)
        else:  # structured
            return self.current_db.add_paper_structured(text, metadata, **kwargs)

    def search(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        """Search using current database"""
        if self.current_db_type == "legacy":
            return self.current_db.search_documents(query, k, **kwargs)
        else:  # structured
            return self.current_db.search_structured(query, k, **kwargs)

    def get_document_count(self) -> int:
        """Get document count"""
        return self.current_db.get_document_count()

    def get_all_papers(self) -> List[Dict]:
        """Get all papers"""
        return self.current_db.get_all_papers()

    def get_stats(self) -> Dict:
        """Get database statistics"""
        stats = {
            "mode": self.mode,
            "database_type": self.current_db_type,
            "document_count": self.get_document_count(),
            "papers_count": len(self.get_all_papers()),
            "available_modes": []
        }

        if LEGACY_AVAILABLE:
            legacy_db = LegacyDatabase()
            stats["legacy_count"] = legacy_db.get_document_count()
            stats["available_modes"].append("legacy")

        if STRUCTURED_AVAILABLE:
            structured_db = StructuredVectorDatabase()
            stats["structured_count"] = structured_db.get_document_count()
            stats["available_modes"].append("structured")

        return stats

    def clear_current_database(self):
        """Clear current database"""
        if hasattr(self.current_db, 'clear_database'):
            self.current_db.clear_database()
        else:
            print("⚠️ Clear method not available for this database")


# ====== STREAMLIT HELPER ======

def create_streamlit_sidebar(db_manager: DatabaseManager):
    """Create Streamlit sidebar for database switching"""
    import streamlit as st

    with st.sidebar:
        st.header("Database Settings")

        # Database mode selector
        available_modes = []
        if LEGACY_AVAILABLE:
            available_modes.append("legacy")
        if STRUCTURED_AVAILABLE:
            available_modes.append("structured")

        if len(available_modes) > 1:
            selected_mode = st.selectbox(
                "Database Mode",
                available_modes,
                index=available_modes.index(db_manager.mode) if db_manager.mode in available_modes else 0
            )

            if selected_mode != db_manager.mode:
                if st.button("Switch Database"):
                    if db_manager.switch_mode(selected_mode):
                        st.success(f"Switched to {selected_mode} database")
                        st.rerun()
                    else:
                        st.error("Failed to switch database")

        # Show stats
        st.subheader("Database Stats")
        stats = db_manager.get_stats()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Mode", stats["database_type"])
            st.metric("Documents", stats["document_count"])

        with col2:
            st.metric("Papers", stats["papers_count"])
            if "legacy_count" in stats:
                st.metric("Legacy Docs", stats["legacy_count"])
            if "structured_count" in stats:
                st.metric("Structured Docs", stats["structured_count"])

        # Clear database button
        st.subheader("Maintenance")
        if st.button("Clear Current Database", type="secondary"):
            if st.checkbox("I understand this will delete all documents", key="clear_confirm"):
                db_manager.clear_current_database()
                st.success("Database cleared!")
                st.rerun()


# ====== QUICK TEST ======
if __name__ == "__main__":
    print("🧪 Testing Database Manager...")

    # Test with auto mode
    manager = DatabaseManager(mode="auto")
    print(f"✅ Manager initialized in {manager.mode} mode")
    print(f"✅ Database type: {manager.current_db_type}")
    print(f"✅ Document count: {manager.get_document_count()}")

    # Get stats
    stats = manager.get_stats()
    print(f"📊 Stats: {stats}")

    # Test switching
    if "structured" in stats["available_modes"]:
        print("\n🔄 Testing switch to structured...")
        if manager.switch_mode("structured"):
            print(f"✅ Switched to {manager.current_db_type}")

    # Test adding and searching
    print("\n📝 Testing add and search...")
    test_result = manager.add_paper(
        "ABSTRACT\nTest abstract for database manager\n\nCONCLUSION\nTest conclusion",
        {"title": "Manager Test", "author": "Test"}
    )
    print(f"✅ Add result: {test_result}")

    search_results = manager.search("test abstract", k=2)
    print(f"✅ Search results: {len(search_results)} documents")