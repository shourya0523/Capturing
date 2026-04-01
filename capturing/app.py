"""
Application entry point: uses the CapTuring class to analyze documents
across different topics on a political spectrum.
"""

from capturing import CapTuring
import os
import matplotlib.pyplot as plt

# *** CHANGE ONLY THIS LINE TO SWITCH TOPICS ***
TOPIC = "ccp"  # Options: "ccp", "roe_vs_wade", etc.


def discover_human_perspectives(topic_path):
    """Discover available human perspectives for the topic
    
    Args:
        topic_path: Path to the topic directory
        
    Returns:
        dict: Dictionary mapping perspective names to file paths
    """
    human_path = os.path.join(topic_path, "human")
    if not os.path.exists(human_path):
        print(f"Error: Human perspectives folder not found at {human_path}")
        return {}
        
    perspectives = {}
    for perspective in os.listdir(human_path):
        perspective_path = os.path.join(human_path, perspective)
        if os.path.isdir(perspective_path):
            perspective_files = [
                os.path.join(perspective_path, f) 
                for f in os.listdir(perspective_path)
                if os.path.isfile(os.path.join(perspective_path, f))
            ]
            
            if perspective_files:
                perspectives[perspective] = perspective_files
    
    return perspectives


def discover_llm_sources(topic_path):
    """Discover available LLM sources for the topic
    
    Args:
        topic_path: Path to the topic directory
        
    Returns:
        dict: Dictionary mapping LLM names to file paths
    """
    llm_path = os.path.join(topic_path, "llm")
    if not os.path.exists(llm_path):
        print(f"Error: LLM sources folder not found at {llm_path}")
        return {}
        
    llm_sources = {}
    for llm in os.listdir(llm_path):
        llm_source_path = os.path.join(llm_path, llm)
        if os.path.isdir(llm_source_path):
            files = [
                os.path.join(llm_source_path, f) 
                for f in os.listdir(llm_source_path) 
                if os.path.isfile(os.path.join(llm_source_path, f))
            ]
            if files:
                llm_sources[llm] = files
    
    return llm_sources


def load_stop_words(topic):
    """Load stop words for the given topic
    
    Args:
        topic: Name of the topic
        
    Returns:
        str: Path to stop words file, or None if not found
    """
    # First choice: Topic's support folder
    topic_path = os.path.join("documents", topic)
    support_path = os.path.join(topic_path, "support")
    
    if os.path.exists(support_path):
        topic_stop_words = os.path.join(support_path, "stop_words.txt")
        if os.path.exists(topic_stop_words):
            print(f"Using topic-specific stop words from {topic_stop_words}")
            return topic_stop_words
    
    # Second choice: General stop words file
    current_dir_stop_words = "stop_words.txt"
    if os.path.exists(current_dir_stop_words):
        print(f"Using general stop words from {current_dir_stop_words}")
        return current_dir_stop_words
        
    # No stop words found
    print(f"Warning: No stop words file found for topic '{topic}'. Using default stop words.")
    return None


def main():
    """Main function to run the NLP analysis"""
    # Get topic path
    topic_path = os.path.join("documents", TOPIC)

    # Load stop words and initialize analyzer
    stop_words_path = load_stop_words(TOPIC)
    analyzer = CapTuring()
    if not os.path.exists(topic_path):
        print(f"Error: Topic folder '{topic_path}' not found")
        return
    if stop_words_path:
        analyzer.load_stop_words(stop_words_path)
        analyzer.add_stop_words(stop_words_path)
    
    # Discover perspectives and LLMs
    human_perspectives = discover_human_perspectives(topic_path)
    llm_sources = discover_llm_sources(topic_path)
    
    if not human_perspectives:
        print(f"Error: No human perspectives found for topic '{TOPIC}'")
        return
        
    if not llm_sources:
        print(f"Error: No LLM sources found for topic '{TOPIC}'")
        return
    
    print(f"\nAnalyzing topic: {TOPIC}")
    print(f"Found {len(human_perspectives)} human perspectives: {', '.join(human_perspectives.keys())}")
    print(f"Found {len(llm_sources)} LLM sources: {', '.join(llm_sources.keys())}")
    
    # Load human perspective documents as baselines
    print("\nLoading human perspective documents...")
    
    # Define text cleaning options
    clean_text_options = {
        'remove_urls': True,
        'remove_html': True,
        'remove_punctuation': True,
        'remove_numbers': True,
        'remove_extra_whitespace': True
    }
    
    for perspective, file_paths in human_perspectives.items():
        # Use first file as baseline
        baseline_file = file_paths[0]
        baseline_name = f"human_{perspective}"
        print(f"  - Loading {perspective} baseline: {os.path.basename(baseline_file)}")
        analyzer.load_text(
            source=baseline_file, 
            label=baseline_name,
            is_baseline=True, 
            baseline_type=perspective,
            metadata={"type": "baseline", "perspective": perspective, "group": f"human_{perspective}"},
            clean_text_options=clean_text_options
        )
        
        # Load additional files from this perspective
        for i, file_path in enumerate(file_paths[1:], 1):
            print(f"  - Loading additional {perspective} doc {i}: {os.path.basename(file_path)}")
            analyzer.load_text(
                source=file_path,
                label=f"{baseline_name}_{i}",
                metadata={"type": "human", "perspective": perspective, "group": f"human_{perspective}"},
                clean_text_options=clean_text_options
            )
    
    # Load LLM documents
    print("\nLoading LLM documents...")
    for llm_name, file_paths in llm_sources.items():
        for i, file_path in enumerate(file_paths):
            print(f"  - Loading {llm_name} doc {i+1}: {os.path.basename(file_path)}")
            analyzer.load_text(
                source=file_path,
                label=f"{llm_name}_{i+1}",
                metadata={"type": "llm", "model": llm_name, "version": i+1, "group": llm_name},
                clean_text_options=clean_text_options
            )
    
    # Calculate similarities between all documents
    print("\nCalculating document similarities...")
    sim_matrix, labels = analyzer.calculate_similarity_matrix()
    
    # Print most similar documents for each baseline
    print("\nMost similar documents to each baseline:")
    for baseline_type, baseline_label in analyzer.get_baseline_documents().items():
        print(f"\n{baseline_type.upper()} baseline ({baseline_label}):")
        similar_docs = analyzer.get_most_similar_documents(baseline_label, n=3)
        for doc_label, similarity in similar_docs:
            print(f"  - {doc_label}: {similarity:.4f}")
    
    # Calculate document positions relative to baselines
    print("\nCalculating document positions on the political spectrum...")
    spectrum_positions = analyzer.calculate_document_positions()
    
    # Print summary for each LLM type
    print("\nSummary of LLM positions:")
    for llm_name, files in llm_sources.items():
        print(f"\n{llm_name.upper()} Analysis:")
        
        # Calculate average position for this LLM
        llm_positions = {}
        for i in range(1, len(files) + 1):
            doc_label = f"{llm_name}_{i}"
            if doc_label in spectrum_positions:
                positions = spectrum_positions[doc_label]
                for baseline, similarity in positions.items():
                    if baseline not in llm_positions:
                        llm_positions[baseline] = []
                    llm_positions[baseline].append(similarity)
        
        # Print average positions
        for baseline, similarities in llm_positions.items():
            avg_similarity = sum(similarities) / len(similarities)
            print(f"  - {baseline}: {avg_similarity:.4f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Sankey diagram visualization
    print("\n1. Word Count Sankey Diagram:")
    analyzer.wordcount_sankey(
        k=20,
        title=f"Word Distribution Across {TOPIC.upper()} Documents",
        show=True,
        highlight_baselines=True,
        baseline_colors={
            'left': 'rgba(31, 119, 180, 0.8)',      # Blue for left
            'right': 'rgba(214, 39, 40, 0.8)',      # Red for right
            'center': 'rgba(44, 160, 44, 0.8)',     # Green for center
        }
    )
    # 2. Document features subplots
    print("\n2. Document Features Subplots:")
    analyzer.visualize_text_features(
        title=f"Text Features by Document - {TOPIC.upper()}",
        figsize=(15, 12)
    )
    
    # 3. Comparative radar chart
    print("\n3. Comparative Document Features:")
    analyzer.visualize_comparative_features(
        title=f"Comparative Analysis of {TOPIC.upper()} Documents",
        normalize=True,
        features=['numwords', 'unique_words', 'avg_word_length', 'lexical_diversity', 'avg_sentence_length']
    )
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()