"""
Quality assurance for processed documents.
"""
from typing import List, Dict, Any, Tuple
from langchain.docstore.document import Document
import logging
import re 
from collections import Counter

logger = logging.getLogger(__name__)

class DocumentQualityValidator: 
    """
    Validates the quality of processed documents.
    
    Industry practice: Automated quality checks ensure data integrity.
    """
    
    def __init__(self):
        self.validation_rules = {
            "min_content_length": 15,  # Reduced from 20
            "max_content_length": 8000,
            "min_quality_score": 0.3,
            "required_fields": ["source", "doc_id", "quality_score"],
            "max_empty_lines_ratio": 0.4  # More lenient
        }
        
        self.validation_stats = {
            "total_validated": 0,
            "passed": 0,
            "failed": 0,
            "warnings": 0
        }

    def validate_content_structure(self, document: Document) -> Tuple[bool, List[str]]:
        """
        Validate document content structure - updated version.
        """
        issues = []
        content = document.page_content
        
        # Check content length
        if len(content) < self.validation_rules["min_content_length"]:
            issues.append(f"Content too short: {len(content)} chars")
        
        if len(content) > self.validation_rules["max_content_length"]:
            issues.append(f"Content too long: {len(content)} chars")
        
        # Check for excessive empty lines
        lines = content.split('\n')
        empty_lines = sum(1 for line in lines if not line.strip())
        if lines and empty_lines / len(lines) > self.validation_rules["max_empty_lines_ratio"]:
            issues.append(f"Too many empty lines: {empty_lines}/{len(lines)}")
        
        # More flexible check for mathematical content structure
        has_problem = "Problem:" in content or "problem" in content.lower()
        has_solution = "Solution:" in content or "solution" in content.lower() or "answer" in content.lower()
        
        if not (has_problem or has_solution):
            issues.append("Missing Problem/Solution structure")
        
        # Check for truncated content
        if content.endswith("..."):
            issues.append("Content appears truncated")
        
        return len(issues) == 0, issues
    
    def validate_metadata(self, document: Document) -> Tuple[bool, List[str]]:
        """
        Validate document metadata completeness and consistency.
        """
        issues = []
        metadata = document.metadata
        
        # Check required fields
        for field in self.validation_rules["required_fields"]:
            if field not in metadata:
                issues.append(f"Missing required metadata field: {field}")
        
        # Check quality score
        quality_score = metadata.get("quality_score")
        if quality_score is not None:
            if quality_score < self.validation_rules["min_quality_score"]:
                issues.append(f"Quality score too low: {quality_score}")
            if not 0 <= quality_score <= 1:
                issues.append(f"Quality score out of range: {quality_score}")
        
        # Check chunk metadata consistency
        if "chunk_index" in metadata and "total_chunks" in metadata:
            chunk_idx = metadata["chunk_index"]
            total_chunks = metadata["total_chunks"]
            if chunk_idx >= total_chunks:
                issues.append(f"Invalid chunk index: {chunk_idx} >= {total_chunks}")
        
        return len(issues) == 0, issues
    
    def validate_mathematical_content(self, document: Document) -> Tuple[bool, List[str]]:
        """
        Validate mathematical content quality - updated version.
        """
        issues = []
        content = document.page_content
        
        # Check for broken LaTeX
        latex_patterns = [r'\$[^$]*\$', r'\$\$[^$]*\$\$']
        for pattern in latex_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if len(match) < 3:  # Too short to be meaningful
                    issues.append(f"Suspicious LaTeX expression: {match}")
        
        # More careful check for code blocks
        if '```' in content:
            code_block_count = content.count('```')
            if code_block_count % 2 != 0:  # Should be even (open/close pairs)
                issues.append("Unmatched code block delimiters")
        
        # More lenient check for mathematical expressions
        open_parens = content.count('(')
        close_parens = content.count(')')
        if abs(open_parens - close_parens) > 2:  # Allow small imbalance
            issues.append("Significantly unbalanced parentheses in mathematical content")
        
        return len(issues) == 0, issues
    
    def validate_document(self, document: Document) -> Dict[str, Any]:
        """
        Comprehensive document validation.
        
        Returns validation result with details.
        """
        result = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "metadata": document.metadata.copy()
        }
        
        # Structure validation
        struct_valid, struct_issues = self.validate_content_structure(document)
        if not struct_valid:
            result["is_valid"] = False
            result["issues"].extend(struct_issues)
        
        # Metadata validation
        meta_valid, meta_issues = self.validate_metadata(document)
        if not meta_valid:
            result["is_valid"] = False
            result["issues"].extend(meta_issues)
        
        # Mathematical content validation
        math_valid, math_issues = self.validate_mathematical_content(document)
        if not math_valid:
            result["warnings"].extend(math_issues)  # These are warnings, not failures
        
        return result
    
    def validate_batch(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Validate a batch of documents and return comprehensive report.
        """
        logger.info(f"Validating batch of {len(documents)} documents...")
        
        valid_documents = []
        invalid_documents = []
        all_issues = []
        all_warnings = []
        
        for doc_idx, document in enumerate(documents):
            validation_result = self.validate_document(document)
            
            if validation_result["is_valid"]:
                valid_documents.append(document)
                self.validation_stats["passed"] += 1
            else:
                invalid_documents.append((doc_idx, document, validation_result))
                self.validation_stats["failed"] += 1
                all_issues.extend(validation_result["issues"])
            
            if validation_result["warnings"]:
                self.validation_stats["warnings"] += 1
                all_warnings.extend(validation_result["warnings"])
            
            self.validation_stats["total_validated"] += 1
        
        # Generate summary report
        report = {
            "total_documents": len(documents),
            "valid_documents": len(valid_documents),
            "invalid_documents": len(invalid_documents),
            "documents_with_warnings": sum(1 for d in documents 
                                         if self.validate_document(d)["warnings"]),
            "validation_rate": len(valid_documents) / len(documents) * 100,
            "common_issues": Counter(all_issues).most_common(5),
            "common_warnings": Counter(all_warnings).most_common(5),
            "valid_docs": valid_documents,
            "invalid_docs": invalid_documents
        }
        
        logger.info(f"Validation complete: {report['valid_documents']}/{report['total_documents']} "
                   f"documents passed ({report['validation_rate']:.1f}%)")
        
        return report
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get cumulative validation statistics."""
        return self.validation_stats.copy()
    
class PipelineTester:
    """
    Automated testing for the entire document processing pipeline.
    
    """
    
    def __init__(self):
        self.test_results = {}
    
    def test_document_processor(self) -> bool:
        """Test document processor functionality."""
        try:
            from src.data.dataset_loader import MathDatasetLoader
            from src.core.document_processor import DocumentProcessor
            
            # Load small sample
            loader = MathDatasetLoader()
            problems = loader.process_examples(limit=5)
            
            # Process documents
            processor = DocumentProcessor()
            documents = processor.process_batch(problems)
            
            # Basic checks
            assert len(documents) > 0, "No documents created"
            assert all(hasattr(doc, 'page_content') for doc in documents), "Invalid document structure"
            assert all(hasattr(doc, 'metadata') for doc in documents), "Missing metadata"
            
            self.test_results["document_processor"] = True
            logger.info("Document processor test: PASSED")
            return True
            
        except Exception as e:
            self.test_results["document_processor"] = False
            logger.error(f"Document processor test: FAILED - {e}")
            return False
    
    def test_text_splitter(self) -> bool:
        """Test text splitter functionality."""
        try:
            from src.data.dataset_loader import MathDatasetLoader
            from src.core.document_processor import DocumentProcessor
            from src.core.text_splitter import MathematicalTextSplitter
            
            # Create test documents
            loader = MathDatasetLoader()
            problems = loader.process_examples(limit=3)
            processor = DocumentProcessor()
            documents = processor.process_batch(problems)
            
            # Test splitting
            splitter = MathematicalTextSplitter(chunk_size=500)
            chunks = splitter.split_documents(documents)
            
            # Basic checks
            assert len(chunks) >= len(documents), "Splitting reduced document count"
            assert all(len(chunk.page_content) <= 1500 for chunk in chunks), "Chunks too large"
            
            self.test_results["text_splitter"] = True
            logger.info("Text splitter test: PASSED")
            return True
            
        except Exception as e:
            self.test_results["text_splitter"] = False
            logger.error(f"Text splitter test: FAILED - {e}")
            return False
    
    def test_quality_validator(self) -> bool:
        """Test quality validation functionality."""
        try:
            from langchain.docstore.document import Document
            
            validator = DocumentQualityValidator()
            
            # Test with good document (more realistic)
            good_doc = Document(
                page_content="Problem: What is the derivative of x^2?\n\nSolution: The derivative is 2x using the power rule.",
                metadata={"source": "test", "doc_id": "test_1", "quality_score": 0.8}
            )
            
            result = validator.validate_document(good_doc)
            assert result["is_valid"], f"Good document failed validation: {result['issues']}"
            
            # Test with bad document
            bad_doc = Document(
                page_content="X",  # Too short
                metadata={"quality_score": 0.1}  # Missing required fields
            )
            
            result = validator.validate_document(bad_doc)
            assert not result["is_valid"], "Bad document passed validation"
            
            self.test_results["quality_validator"] = True
            logger.info("Quality validator test: PASSED")
            return True
            
        except Exception as e:
            self.test_results["quality_validator"] = False
            logger.error(f"Quality validator test: FAILED - {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all pipeline tests."""
        logger.info("Running comprehensive pipeline tests...")
        
        tests = [
            ("Document Processor", self.test_document_processor),
            ("Text Splitter", self.test_text_splitter),
            ("Quality Validator", self.test_quality_validator)
        ]
        
        for test_name, test_func in tests:
            logger.info(f"Running {test_name} test...")
            test_func()
        
        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)
        
        logger.info(f"Pipeline tests complete: {passed}/{total} passed")
        return self.test_results