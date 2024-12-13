import random
import string
import unittest

from interfaces.utils import text_to_chunks

def create_documents(documents_topology):
  documents = []
  for topo in documents_topology:
    text = [random.choice(string.ascii_letters + ' ') for i in range(sum(topo))]
    documents.append(text)

  return documents

class TestTextToChunk(unittest.TestCase):
    def test_case_1(self):
        """Test case 1 with MODEL_MAX_CHAR_SIZE = 200"""
        documents_topology = [[200, 111], [80], [96], [100], [170], [250]]
        documents = create_documents(documents_topology)
        
        chunks, document_chunk_indices, super_chunk_indices = text_to_chunks(documents, chunk_size=200, batch_size=1)
        
        expected_doc_indices = [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 8)]
        expected_super_indices = [(0, 1), (1, 3), (3, 5), (5, 6), (6, 7), (7, 8)]
        
        self.assertEqual(document_chunk_indices, expected_doc_indices)
        self.assertEqual(super_chunk_indices, expected_super_indices)

    def test_case_2(self):
        """Test case 2 with MODEL_MAX_CHAR_SIZE = 200"""
        documents_topology = [[200, 111], [80], [96], [100], [170], [31]]
        documents = create_documents(documents_topology)
        
        chunks, document_chunk_indices, super_chunk_indices = text_to_chunks(documents, chunk_size=200, batch_size=1)
        
        expected_doc_indices = [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
        expected_super_indices = [(0, 1), (1, 3), (3, 5), (5, 6), (6, 7)]
        
        self.assertEqual(document_chunk_indices, expected_doc_indices)
        self.assertEqual(super_chunk_indices, expected_super_indices)

    def test_case_3(self):
        """Test case 3 with MODEL_MAX_CHAR_SIZE = 200"""
        documents_topology = [[200, 111], [80], [96], [100], [169], [31]]
        documents = create_documents(documents_topology)
        
        chunks, document_chunk_indices, super_chunk_indices = text_to_chunks(documents, chunk_size=200, batch_size=1)
        
        expected_doc_indices = [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
        expected_super_indices = [(0, 1), (1, 3), (3, 5), (5, 7)]
        
        self.assertEqual(document_chunk_indices, expected_doc_indices)
        self.assertEqual(super_chunk_indices, expected_super_indices)

    def test_case_4(self):
        """Test case 4 with MODEL_MAX_CHAR_SIZE = 60"""
        documents_topology = [[200, 111], [80], [96], [100], [169], [31]]
        documents = create_documents(documents_topology)
        
        chunks, document_chunk_indices, super_chunk_indices = text_to_chunks(documents, chunk_size=60, batch_size=1)
        
        expected_doc_indices = [(0, 6), (6, 8), (8, 10), (10, 12), (12, 15), (15, 16)]
        expected_super_indices = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), 
            (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
            (12, 13), (13, 14), (14, 15), (15, 16)
        ]
        
        self.assertEqual(document_chunk_indices, expected_doc_indices)
        self.assertEqual(super_chunk_indices, expected_super_indices)

if __name__ == '__main__':
  unittest.main()