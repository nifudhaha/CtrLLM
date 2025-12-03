

from descriptor import TextDescriptor


def main():
    # Initialize descriptor
    print("Initializing Text Descriptor...")
    descriptor = TextDescriptor()
    
    # Example text about a controversial topic
    text = """
    Climate change represents one of the most significant challenges facing humanity today. 
    Scientists worldwide, including experts from NASA, NOAA, and the IPCC, agree that global 
    temperatures are rising at an unprecedented rate due to greenhouse gas emissions from 
    human activities.
    
    However, debates continue about the best policy approaches to address this crisis. 
    Some environmental advocates argue for immediate and dramatic action, including rapid 
    transitions away from fossil fuels. They believe that incremental changes are insufficient 
    given the urgency of the situation.
    
    Others, including certain economists and industry representatives, prefer more gradual 
    transitions. They worry that overly aggressive policies could harm economic growth and 
    disproportionately affect vulnerable communities. These groups emphasize the need for 
    balanced approaches that consider both environmental and economic factors.
    
    Meanwhile, developing nations face unique challenges. They argue that developed countries, 
    which contributed most to historical emissions, should bear greater responsibility for 
    mitigation efforts. This raises complex questions about climate justice and international 
    cooperation.
    
    The scientific consensus is clear on the reality and severity of climate change. Yet 
    translating this scientific agreement into effective policy remains contentious, involving 
    trade-offs between competing values and interests.
    """
    
    print("\n" + "="*60)
    print("COMPUTING METRICS")
    print("="*60)
    
    # Compute all metrics
    print("\nAnalyzing text...")
    results = descriptor.compute_all_metrics(
        text,
        include_sentiment=True,
        include_embeddings=True,
        include_entities=True
    )
    
    # Display summary
    descriptor.print_summary(results)
    
    # Save results
    output_file = "demo_metrics.pkl"
    descriptor.save_results(results, output_file)
    
    # Example: Accessing specific metrics
    print("\n" + "="*60)
    print("ACCESSING SPECIFIC METRICS")
    print("="*60)
    
    print(f"\n1. Shannon Entropy:")
    print(f"   - Entropy value: {results['shannon_entropy']['entropy']:.3f}")
   
    print(f"\n2. Lexical Ratio:")
    print(f"   - Content words: {results['pos_features']['lexical_ratio']:.1%}")
   
    
    print(f"\n3. Entity Specificity:")
    print(f"   - Specificity score: {results['entity_specificity']['entity_specificity']:.2f}")
    print(f"   - Unique entities: {results['entity_specificity']['unique_entities']}")
    print(f"   - Entity breakdown: {results['entity_specificity']['person_count']} persons, "
          f"{results['entity_specificity']['org_count']} organizations, "
          f"{results['entity_specificity']['gpe_count']} locations")
    
    print(f"\n4. Sentiment Variance:")
    print(f"   - Variance: {results['sentiment_variance']['sentiment_variance']:.3f}")

    
    print(f"\n5. Embedding Variance:")
    print(f"   - Variance: {results['embedding_variance']:.3f}")




if __name__ == "__main__":
    main()
