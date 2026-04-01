"""
File: capturing.py

Description: A streamlined framework for comparative text analysis
designed to analyze and visualize relationships between documents.
"""

from collections import Counter, defaultdict
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os


class CapTuring:
    """
    CapTuring: A framework for comparative text analysis.
    
    This class provides methods to load, analyze, and visualize text documents,
    with a focus on comparison between different perspectives.
    """

    def __init__(self, vectorizer=None, **kwargs):
        """Constructor"""
        self.data = defaultdict(dict)
        self.documents = {}  # Store raw text of documents
        self.baselines = {}  # Store baseline documents
        
        # Configure vectorizer
        vectorizer_kwargs = kwargs.get('vectorizer_kwargs', {})
        self.vectorizer = vectorizer or TfidfVectorizer(stop_words='english', **vectorizer_kwargs)
        self.parsers = {'simple': self.simple_text_parser}
        self.tfidf_matrix = None
        self.similarity_matrix = None
        
        # Configuration
        self.config = {
            'similarity_metric': kwargs.get('similarity_metric', 'cosine'),
            'min_word_length': kwargs.get('min_word_length', 2),
            'case_sensitive': kwargs.get('case_sensitive', False)
        }

    def simple_text_parser(self, filename):
        """Parse a simple text file"""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                text = file.read()
            words = self.tokenize_text(text)
            word_count = Counter(words)
            
            results = {
                'text': text,
                'wordcount': word_count,
                'numwords': len(words),
                'unique_words': len(word_count)
            }
            
            results.update(self.calculate_advanced_features(text, words))
            return results
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            return {'text': '', 'wordcount': Counter(), 'numwords': 0, 'unique_words': 0}

    def tokenize_text(self, text):
        """Tokenize text into words"""
        if not self.config['case_sensitive']:
            text = text.lower()
            
        min_length = self.config['min_word_length']
        words = re.findall(rf'\b\w{{{min_length},}}\b', text)
        
        # Filter out stop words
        if hasattr(self.vectorizer, 'stop_words_'):
            words = [word for word in words if word not in self.vectorizer.stop_words_]
            
        return words
    
    def clean_text(self, text, **kwargs):
        """Clean and normalize text data"""
        # Remove URLs
        if kwargs.get('remove_urls', True):
            text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        
        # Remove HTML tags
        if kwargs.get('remove_html', True):
            text = re.sub(r'<.*?>', ' ', text)
            
        # Remove punctuation
        if kwargs.get('remove_punctuation', False):
            text = re.sub(r'[^\w\s]', ' ', text)
            
        # Remove numbers
        if kwargs.get('remove_numbers', False):
            text = re.sub(r'\d+', ' ', text)
            
        # Normalize whitespace
        if kwargs.get('remove_extra_whitespace', True):
            text = re.sub(r'\s+', ' ', text).strip()
            
        return text
    
    def load_text(self, source, label=None, **kwargs):
        """Register a document with the framework"""
        # Extract common kwargs
        is_baseline = kwargs.get('is_baseline', False)
        baseline_type = kwargs.get('baseline_type')
        metadata = kwargs.get('metadata', {})
        clean_text_options = kwargs.get('clean_text_options', {})
        
        # Determine source type and generate label
        is_file = isinstance(source, str) and (
            source.endswith('.txt') or source.endswith('.md') or 
            source.endswith('.html') or '/' in source or '\\' in source
        )
        
        if label is None:
            label = kwargs.get('label_prefix', '') + source if is_file else f"doc_{len(self.documents) + 1}"
        
        if is_file:
            parser_name = kwargs.get('parser_name', 'simple')
            parser_func = self.parsers.get(parser_name, self.simple_text_parser)
            results = parser_func(source)
            
            # Clean the text
            if 'text' in results:
                results['text'] = self.clean_text(results['text'], **clean_text_options)
                words = self.tokenize_text(results['text'])
                results['wordcount'] = Counter(words)
                results['numwords'] = len(words)
                results['unique_words'] = len(results['wordcount'])
                results.update(self.calculate_advanced_features(results['text'], words))
        else:
            text = self.clean_text(source, **clean_text_options)
            words = self.tokenize_text(text)
            
            results = {
                'text': text,
                'wordcount': Counter(words),
                'numwords': len(words),
                'unique_words': len(set(words))
            }
            results.update(self.calculate_advanced_features(text, words))

        # Store document and data
        self.documents[label] = results['text']
        
        for k, v in results.items():
            self.data[k][label] = v
            
        if metadata:
            if 'metadata' not in self.data:
                self.data['metadata'] = {}
            self.data['metadata'][label] = metadata
            
        # Register as baseline
        if is_baseline and baseline_type:
            self.baselines[baseline_type] = label
            print(f"Registered '{label}' as {baseline_type} baseline")

        # Reset matrices
        self.tfidf_matrix = None
        self.similarity_matrix = None
        
        return label

    def add_stop_words(self, stop_words_list):
        """Add custom stop words to the vectorizer"""
        if not stop_words_list:
            return self
            
        # Initialize stop_words_ attribute if needed
        if not hasattr(self.vectorizer, 'stop_words_'):
            self.vectorizer.stop_words_ = set() if not hasattr(self.vectorizer, 'get_stop_words') else set(self.vectorizer.get_stop_words())
                
        # Add new stop words
        self.vectorizer.stop_words_.update(stop_words_list)
        return self
        
    def load_stop_words(self, stopfile, **kwargs):
        """Load custom stop words from a file"""
        verbose = kwargs.get('verbose', True)
        stop_words = []
        
        try:
            with open(stopfile, 'r', encoding='utf-8') as file:
                for line in file:
                    word = line.strip()
                    if word and not word.startswith('#'):
                        stop_words.append(word)
            
            self.add_stop_words(stop_words)
            
            if verbose:
                print(f"Loaded {len(stop_words)} stop words from {stopfile}")
            return stop_words
        except Exception as e:
            if verbose:
                print(f"Error loading stop words from {stopfile}: {e}")
            return []

    def calculate_similarity_matrix(self):
        """Calculate cosine similarity between all documents"""
        if not self.documents:
            return None, []
            
        # Create TF-IDF matrix if needed
        if self.tfidf_matrix is None:
            labels = list(self.documents.keys())
            texts = [self.documents[label] for label in labels]
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        else:
            labels = list(self.documents.keys())
        
        # Calculate similarity matrix if needed
        if self.similarity_matrix is None:
            self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
            
        return self.similarity_matrix, labels

    def get_document_similarity(self, doc1_label, doc2_label):
        """Get similarity between two documents"""
        sim_matrix, labels = self.calculate_similarity_matrix()
        
        if sim_matrix is None:
            return None
            
        try:
            idx1 = labels.index(doc1_label)
            idx2 = labels.index(doc2_label)
            return sim_matrix[idx1, idx2]
        except ValueError:
            print(f"Document labels not found")
            return None

    def get_most_similar_documents(self, doc_label, n=5):
        """Find most similar documents to a given document"""
        sim_matrix, labels = self.calculate_similarity_matrix()
        
        if sim_matrix is None or doc_label not in labels:
            return []
            
        idx = labels.index(doc_label)
        similarities = sim_matrix[idx]
        
        # Create list of (label, similarity) tuples and sort
        similar_docs = [(labels[i], similarities[i]) for i in range(len(labels))]
        similar_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Remove self from results
        similar_docs = [doc for doc in similar_docs if doc[0] != doc_label]
                
        return similar_docs[:n]
    def get_baseline_documents(self, **kwargs):
        """Get all baseline documents
        
        Args:
            **kwargs: Additional options including:
                - types_only: Return only baseline types (default: False)
                - include_text: Include document text (default: False)
            
        Returns:
            dict: Dictionary mapping baseline types to document labels or data
        """
        if kwargs.get('types_only', False):
            return list(self.baselines.keys())
            
        if kwargs.get('include_text', False):
            result = {}
            for btype, label in self.baselines.items():
                result[btype] = {
                    'label': label,
                    'text': self.get_document_text(label)
                }
            return result
            
        return self.baselines.copy()
    def calculate_document_positions(self, baseline_types=None):
        """Calculate document positions relative to baselines"""
        # Use all baselines if none specified
        if baseline_types is None:
            baseline_types = list(self.baselines.keys())
            
        if not baseline_types:
            print("No baseline documents specified")
            return {}
            
        sim_matrix, labels = self.calculate_similarity_matrix()
        if sim_matrix is None:
            return {}
            
        positions = {}
        
        # Get indices of baseline documents
        baseline_indices = {
            btype: labels.index(self.baselines[btype]) 
            for btype in baseline_types if btype in self.baselines
            and self.baselines[btype] in labels
        }
        
        # Calculate positions
        for i, label in enumerate(labels):
            doc_positions = {
                btype: sim_matrix[i, idx] for btype, idx in baseline_indices.items()
            }
            positions[label] = doc_positions
            
        return positions

    def calculate_advanced_features(self, text, words):
        """Calculate advanced text features"""
        features = {}
        
        # Calculate average word length
        features['avg_word_length'] = sum(len(word) for word in words) / len(words) if words else 0
            
        # Calculate lexical diversity
        features['lexical_diversity'] = len(set(words)) / len(words) if words else 0
            
        # Calculate average sentence length
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        if sentences:
            sentence_lengths = [len(re.findall(r'\b\w+\b', s.lower())) for s in sentences]
            features['avg_sentence_length'] = sum(sentence_lengths) / len(sentences)
            features['num_sentences'] = len(sentences)
        else:
            features['avg_sentence_length'] = 0
            features['num_sentences'] = 0
            
        return features

    def visualize_text_features(self, **kwargs):
        """Create a visualization showing each document's features or similarity to baselines"""
            
        # Get document labels - default to all non-baseline documents
        doc_labels = kwargs.get('document_labels')
        if doc_labels is None:
                baseline_labels = set(self.baselines.values())
                doc_labels = [label for label in self.documents if label not in baseline_labels]
            
        # Get baseline types for similarity visualization
        baseline_types = kwargs.get('baseline_types')
        show_similarities = kwargs.get('show_similarities', True)
        
        # Calculate similarities only once if needed
        positions = None
        if show_similarities:
            if baseline_types is None:
                baseline_types = list(self.baselines.keys())
                
            # Calculate similarities to baselines
            positions = self.calculate_document_positions(baseline_types)
            
            # Set features to baseline similarities
            features = baseline_types
            title = kwargs.get('title', 'Document Similarity to Baselines')
        else:
            # Define features to display
            default_features = ['numwords', 'unique_words', 'avg_word_length', 'lexical_diversity']
            features = kwargs.get('features', default_features)
            title = kwargs.get('title', 'Document Text Features')
        
        # Set up figure
        n_docs = len(doc_labels)
        n_cols = min(3, n_docs)
        n_rows = (n_docs + n_cols - 1) // n_cols
        
        figsize = kwargs.get('figsize', (15, 5 * n_rows))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Handle single row/column cases
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
            
        # Create bar charts for each document
        for i, label in enumerate(doc_labels):
            row = i // n_cols
            col = i % n_cols
            
            ax = axes[row, col]
            
            if show_similarities and label in positions:
                # Get similarity values for this document
                feature_values = [positions[label].get(feature, 0) for feature in features]
                feature_labels = [feature.replace('_', ' ').title() for feature in features]
            else:
                # Get document features
                doc_features = {}
                for k, v in self.data.items():
                    if label in v:
                        doc_features[k] = v[label]
                
                # Extract values for selected features
                feature_values = []
                feature_labels = []
                
                for feature in features:
                    if feature in doc_features:
                        feature_values.append(doc_features[feature])
                        feature_labels.append(feature.replace('_', ' ').title())
                    
            # Create bar chart
            x = np.arange(len(feature_values))
            bars = ax.bar(x, feature_values, width=0.6)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
            
            # Set labels and title
            ax.set_xticks(x)
            ax.set_xticklabels(feature_labels, rotation=45, ha='right')
            ax.set_title(label)
            
            # Adjust y-axis to start from 0
            ax.set_ylim(bottom=0)
            
        # Hide empty subplots
        for i in range(n_docs, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])
            
        # Set overall title
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        # Show figure
        if kwargs.get('show', True):
            plt.show()
            
        return fig

    def visualize_comparative_features(self, **kwargs):
        """Create a radar chart comparing documents"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("This visualization requires matplotlib. Install with: pip install matplotlib")
            return None
            
        # Get document labels
        doc_labels = kwargs.get('document_labels', list(self.documents.keys()))
        
        # Define features to compare
        default_features = [
            'numwords', 'unique_words', 'avg_word_length', 'lexical_diversity', 'avg_sentence_length'
        ]
        features = kwargs.get('features', default_features)
        
        # Set up radar chart
        figsize = kwargs.get('figsize', (12, 8))
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        # Collect data more efficiently
        data = []
        for label in doc_labels:
            # Extract feature values directly
            feature_values = []
            for feature in features:
                if feature in self.data and label in self.data[feature]:
                    feature_values.append(self.data[feature][label])
                else:
                    feature_values.append(0)  # Default value if feature not found
            data.append(feature_values)
            
            
        # Convert to numpy array and normalize
        data = np.array(data)
        
        if kwargs.get('normalize', True):
            max_vals = np.max(data, axis=0)
            max_vals[max_vals == 0] = 1
            data = data / max_vals
            
        # Set up angles for radar chart
        angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
        angles += angles[:1]  # Close the circle
        data = np.column_stack((data, data[:, 0]))  # Close the data loop
        
        # Feature labels
        feature_labels = [f.replace('_', ' ').title() for f in features]
        
        # Plot each document
        for i, label in enumerate(doc_labels):
            ax.plot(angles, data[i], linewidth=2, linestyle='solid', label=label)
            ax.fill(angles, data[i], alpha=0.1)
            
        # Configure axes
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_labels)
        
        ax.set_rlabel_position(0)
        if kwargs.get('normalize', True):
            ax.set_yticks([0.25, 0.5, 0.75, 1])
            ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"])
        
        # Add title and legend
        title = kwargs.get('title', 'Comparative Document Features')
        ax.set_title(title)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # Show figure
        if kwargs.get('show', True):
            plt.tight_layout()
            plt.show()
            
        return fig

    def wordcount_sankey(self, word_list=None, k=5, **kwargs):
        """Create a Sankey diagram showing word flows between documents
        
        Args:
            word_list: List of specific words to include (default: None, uses words with count >= k)
            k: Minimum word count to include a word (default: 5)
            **kwargs: Additional options for customization
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("This visualization requires plotly. Install with: pip install plotly")
            return None
            
        # Get document labels
        doc_labels = kwargs.get('document_labels', list(self.documents.keys()))
        
        # Get word counts directly from the data structure
        exclude_words = kwargs.get('exclude_words', [])
        word_counts = {}
        
        # Only process documents that have word counts
        if 'wordcount' in self.data:
            for label in doc_labels:
                if label in self.data['wordcount']:
                    # Get existing word counts and filter excluded words
                    counts = {word: count for word, count in self.data['wordcount'][label].items() 
                             if word not in exclude_words}
                    word_counts[label] = counts
                
        # Build word list from words with count >= k if not provided
        if word_list is None:
            word_list = set()
            for label, counts in word_counts.items():
                # Get words with count >= k
                filtered_words = [word for word, count in counts.items() if count >= k]
                word_list.update(filtered_words)
            word_list = list(word_list)
        
        # Create Sankey data
        sources = []
        targets = []
        values = []
        labels = []
        
        # Add document and word nodes
        doc_indices = {label: i for i, label in enumerate(doc_labels)}
        word_indices = {word: i + len(doc_labels) for i, word in enumerate(word_list)}
        
        labels = list(doc_labels) + list(word_list)
        
        # Add links
        for doc_label, counts in word_counts.items():
            doc_idx = doc_indices[doc_label]
            
            for word in word_list:
                if word in counts:
                    word_idx = word_indices[word]
                    count = counts[word]
                    
                    sources.append(doc_idx)
                    targets.append(word_idx)
                    values.append(count)
        
        # Set node colors
        node_colors = []
        highlight_baselines = kwargs.get('highlight_baselines', False)
        
        if highlight_baselines:
            # Define default colors for baseline types
            baseline_colors = kwargs.get('baseline_colors', {})
            
            for label in labels:
                if label in self.baselines.values():
                    # Find baseline type
                    baseline_type = None
                    for btype, blabel in self.baselines.items():
                        if blabel == label:
                            baseline_type = btype
                            break
                    
                    # Set color based on baseline type
                    color = baseline_colors.get(baseline_type, 'rgba(128, 128, 128, 0.5)')
                    node_colors.append(color)
                else:
                    node_colors.append('rgba(128, 128, 128, 0.5)')  # Gray
        else:
            # Default colors - documents blue, words orange
            for i in range(len(labels)):
                if i < len(doc_labels):
                    node_colors.append('rgba(31, 119, 180, 0.8)')  # Blue for documents
                else:
                    node_colors.append('rgba(255, 127, 14, 0.8)')  # Orange for words
        
        link_colors = []
        link_colors = [node_colors[source] for source in sources]
        # Create figure
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=node_colors
                
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors
            )
        )])
        
        # Configure layout
        width = kwargs.get('width', 1200)
        height = kwargs.get('height', 800)
        title = kwargs.get('title', 'Word Distribution Across Documents')
        
        fig.update_layout(
            title_text=title,
            font_size=12,
            width=width,
            height=height
        )
        
        # Show figure
        if kwargs.get('show', True):
            renderer = kwargs.get('renderer', 'browser')
            fig.show(renderer=renderer)
        
        return fig