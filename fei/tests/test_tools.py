#!/usr/bin/env python3
"""
Tests for Fei code tools
"""

import os
import tempfile
import unittest

from fei.tools.code import (
    GlobFinder,
    GrepTool,
    CodeEditor,
    FileViewer,
    DirectoryExplorer
)

class TestGlobFinder(unittest.TestCase):
    """Tests for GlobFinder"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.finder = GlobFinder(self.temp_dir.name)
        
        # Create test files
        self.create_test_file("test1.py", "print('test1')")
        self.create_test_file("test2.py", "print('test2')")
        self.create_test_file("subdir/test3.py", "print('test3')")
        self.create_test_file("test.txt", "This is a text file")
    
    def tearDown(self):
        """Clean up test environment"""
        self.temp_dir.cleanup()
    
    def create_test_file(self, path, content):
        """Create a test file"""
        full_path = os.path.join(self.temp_dir.name, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
    
    def test_find(self):
        """Test find method"""
        # Find Python files
        py_files = self.finder.find("**/*.py")
        self.assertEqual(len(py_files), 3)
        
        # Find text files
        txt_files = self.finder.find("**/*.txt")
        self.assertEqual(len(txt_files), 1)
        
        # Find in specific directory
        subdir_files = self.finder.find("**/*.py", os.path.join(self.temp_dir.name, "subdir"))
        self.assertEqual(len(subdir_files), 1)
    
    def test_find_with_ignore(self):
        """Test find_with_ignore method"""
        # Find Python files, ignoring subdir
        py_files = self.finder.find_with_ignore("**/*.py", ["subdir"])
        self.assertEqual(len(py_files), 2)


class TestGrepTool(unittest.TestCase):
    """Tests for GrepTool"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.finder = GlobFinder(self.temp_dir.name)
        self.grep = GrepTool(self.temp_dir.name, self.finder)
        
        # Create test files
        self.create_test_file("test1.py", "print('test1')\nprint('hello')")
        self.create_test_file("test2.py", "print('test2')\nprint('world')")
        self.create_test_file("subdir/test3.py", "print('test3')\nprint('hello world')")
    
    def tearDown(self):
        """Clean up test environment"""
        self.temp_dir.cleanup()
    
    def create_test_file(self, path, content):
        """Create a test file"""
        full_path = os.path.join(self.temp_dir.name, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
    
    def test_search(self):
        """Test search method"""
        # Search for 'hello'
        results = self.grep.search("hello")
        self.assertEqual(len(results), 2)
        
        # Search for 'test' in Python files
        results = self.grep.search("test", "**/*.py")
        self.assertEqual(len(results), 3)
        
        # Search for 'world' in specific directory
        results = self.grep.search("world", path=os.path.join(self.temp_dir.name, "subdir"))
        self.assertEqual(len(results), 1)
    
    def test_search_single_file(self):
        """Test search_single_file method"""
        file_path = os.path.join(self.temp_dir.name, "test1.py")
        
        # Search for 'test1'
        results = self.grep.search_single_file(file_path, "test1")
        self.assertEqual(len(results), 1)
        
        # Search for non-existent pattern
        results = self.grep.search_single_file(file_path, "nonexistent")
        self.assertEqual(len(results), 0)


class TestCodeEditor(unittest.TestCase):
    """Tests for CodeEditor"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.editor = CodeEditor(backup=False)
        
        # Create test file
        self.test_file = os.path.join(self.temp_dir.name, "test.py")
        with open(self.test_file, "w") as f:
            f.write("def test():\n    print('test')\n    return 0\n")
    
    def tearDown(self):
        """Clean up test environment"""
        self.temp_dir.cleanup()
    
    def test_edit_file(self):
        """Test edit_file method"""
        # Edit file
        success, message = self.editor.edit_file(
            self.test_file,
            "    print('test')",
            "    print('edited')"
        )
        
        self.assertTrue(success)
        
        # Verify edit
        with open(self.test_file, "r") as f:
            content = f.read()
        
        self.assertIn("print('edited')", content)
    
    def test_create_file(self):
        """Test create_file method"""
        # Create file
        new_file = os.path.join(self.temp_dir.name, "new.py")
        success, message = self.editor.create_file(new_file, "print('new')")
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(new_file))
        
        # Verify content
        with open(new_file, "r") as f:
            content = f.read()
        
        self.assertEqual(content, "print('new')")


class TestFileViewer(unittest.TestCase):
    """Tests for FileViewer"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.viewer = FileViewer()
        
        # Create test file
        self.test_file = os.path.join(self.temp_dir.name, "test.py")
        with open(self.test_file, "w") as f:
            f.write("line1\nline2\nline3\nline4\nline5\n")
    
    def tearDown(self):
        """Clean up test environment"""
        self.temp_dir.cleanup()
    
    def test_view(self):
        """Test view method"""
        # View entire file
        success, message, lines = self.viewer.view(self.test_file)
        
        self.assertTrue(success)
        self.assertEqual(len(lines), 5)
        
        # View with limit
        success, message, lines = self.viewer.view(self.test_file, limit=2)
        
        self.assertTrue(success)
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], "line1")
        
        # View with offset
        success, message, lines = self.viewer.view(self.test_file, offset=2)
        
        self.assertTrue(success)
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0], "line3")


class TestDirectoryExplorer(unittest.TestCase):
    """Tests for DirectoryExplorer"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.explorer = DirectoryExplorer()
        
        # Create test files and directories
        os.makedirs(os.path.join(self.temp_dir.name, "subdir"))
        open(os.path.join(self.temp_dir.name, "file1.txt"), "w").close()
        open(os.path.join(self.temp_dir.name, "file2.py"), "w").close()
    
    def tearDown(self):
        """Clean up test environment"""
        self.temp_dir.cleanup()
    
    def test_list_directory(self):
        """Test list_directory method"""
        # List directory
        success, message, content = self.explorer.list_directory(self.temp_dir.name)
        
        self.assertTrue(success)
        self.assertEqual(len(content["dirs"]), 1)
        self.assertEqual(content["dirs"][0], "subdir")
        self.assertEqual(len(content["files"]), 2)
        
        # List with ignore
        success, message, content = self.explorer.list_directory(
            self.temp_dir.name,
            ignore=["*.txt"]
        )
        
        self.assertTrue(success)
        self.assertEqual(len(content["files"]), 1)
        self.assertEqual(content["files"][0], "file2.py")


if __name__ == "__main__":
    unittest.main()