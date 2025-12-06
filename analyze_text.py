#!/usr/bin/env python3
"""
Docker-compatible command-line tool for comprehensive linguistic analysis of Russian texts.
Generates an HTML report with lexical diversity metrics, morphological analysis, and visualizations.

Usage inside Docker:
    docker run linguistic-analyzer /app/input/text.txt -o /app/output/report.html
"""

import sys
import argparse
import os
import re
import json
from collections import Counter
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Docker
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk
from pymystem3 import Mystem
from lexical_diversity import lex_div as ld
from ruts import BasicStats, SentsExtractor, WordsExtractor
import spacy
from pymorphy2 import MorphAnalyzer

# Configure matplotlib for Docker environment
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.family': 'DejaVu Sans',  # Available in Docker container
    'axes.unicode_minus': False
})

# Download required NLTK data
nltk_data_path = '/app/nltk_data'
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

try:
    nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
    nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
except Exception as e:
    print(f"⚠️ Warning: Could not download NLTK  {e}", file=sys.stderr)

# Initialize analyzers
morph = MorphAnalyzer()
nlp = spacy.load("ru_core_news_sm")

def clean_text(text):
    """Clean text from special characters"""
    text = re.sub(r'[^\w\sёЁа-яА-Я.,!?;:()\-\—]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def morphological_analysis(text):
    """Perform morphological analysis using pymorphy2"""
    words = nltk.word_tokenize(text.lower(), language='russian')
    stopwords = set(nltk.corpus.stopwords.words('russian'))
    content_words = [w for w in words if w.isalpha() and len(w) > 2 and w not in stopwords]
    
    if not content_words:
        return {}
    
    # Analyze parts of speech
    pos_counts = Counter()
    animacy_counts = Counter()
    gender_counts = Counter()
    number_counts = Counter()
    case_counts = Counter()
    tense_counts = Counter()
    
    for word in content_words:
        parses = morph.parse(word)
        if parses:
            parse = parses[0]  # Take most probable parse
            pos = parse.tag.POS
            
            if pos:
                pos_counts[pos] += 1
                
                # Animacy for nouns
                if pos == 'NOUN' and parse.tag.animacy:
                    animacy_counts[parse.tag.animacy] += 1
                
                # Gender for nouns and adjectives
                if pos in ['NOUN', 'ADJF'] and parse.tag.gender:
                    gender_counts[parse.tag.gender] += 1
                
                # Number for nouns, adjectives, verbs
                if pos in ['NOUN', 'ADJF', 'VERB'] and parse.tag.number:
                    number_counts[parse.tag.number] += 1
                
                # Case for nouns and adjectives
                if pos in ['NOUN', 'ADJF'] and parse.tag.case:
                    case_counts[parse.tag.case] += 1
                
                # Tense for verbs
                if pos == 'VERB' and parse.tag.tense:
                    tense_counts[parse.tag.tense] += 1
    
    # Normalize by total content words
    total = len(content_words)
    pos_percentages = {pos: count/total*100 for pos, count in pos_counts.items()}
    
    return {
        'pos_counts': pos_counts,
        'pos_percentages': pos_percentages,
        'animacy_counts': animacy_counts,
        'gender_counts': gender_counts,
        'number_counts': number_counts,
        'case_counts': case_counts,
        'tense_counts': tense_counts,
        'total_content_words': total
    }

def syntactic_complexity(text):
    """Analyze syntactic complexity"""
    doc = nlp(text)
    
    # Sentence lengths
    sent_lengths = [len([token for token in sent if not token.is_punct]) for sent in doc.sents]
    
    # Complexity markers
    clause_markers = [',', ';', ':', 'а', 'но', 'и', 'потому что', 'который', 'что', 'если']
    clause_counts = []
    
    for sent in doc.sents:
        text = sent.text.lower()
        count = sum(1 for marker in clause_markers if marker in text)
        clause_counts.append(count)
    
    # Average sentence length
    avg_sent_length = np.mean(sent_lengths) if sent_lengths else 0
    
    # Sentence complexity index
    avg_clause_count = np.mean(clause_counts) if clause_counts else 0
    
    return {
        'sentence_lengths': sent_lengths,
        'avg_sentence_length': avg_sent_length,
        'clause_counts': clause_counts,
        'avg_clause_count': avg_clause_count,
        'total_sentences': len(sent_lengths)
    }

def readability_index(text):
    """Calculate readability index for Russian text"""
    words = nltk.word_tokenize(text.lower(), language='russian')
    sentences = nltk.sent_tokenize(text, language='russian')
    
    # Words longer than 6 characters are considered complex
    long_words = [w for w in words if len(w) > 6 and w.isalpha()]
    
    avg_word_length = np.mean([len(w) for w in words if w.isalpha()]) if words else 0
    avg_sent_length = len(words) / len(sentences) if sentences else 0
    
    # Simplified readability index adapted from Flesch-Kincaid
    readability = 206.835 - 1.3 * avg_word_length - 60.1 * (len(sentences) / len(words)) if words and sentences else 0
    
    return {
        'readability_score': max(0, min(100, readability)),  # Clamp to 0-100
        'avg_word_length': avg_word_length,
        'avg_sentence_length': avg_sent_length,
        'percent_long_words': len(long_words) / len(words) * 100 if words else 0
    }

def generate_visualizations(text, words, morph_analysis, output_dir):
    """Generate visualization images and return their paths"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Word length distribution
    plt.figure(figsize=(10, 4))
    word_lengths = [len(w) for w in words if w.isalpha()]
    if word_lengths:
        sns.histplot(word_lengths, bins=range(1, max(word_lengths)+2), kde=True)
        plt.title('Distribution of Word Lengths (in characters)')
        plt.xlabel('Word Length')
        plt.ylabel('Frequency')
        word_lengths_path = os.path.join(output_dir, 'word_lengths.png')
        plt.savefig(word_lengths_path, bbox_inches='tight')
        plt.close()
    else:
        word_lengths_path = None
    
    # Parts of speech distribution
    pos_path = None
    if morph_analysis and morph_analysis.get('pos_percentages'):
        plt.figure(figsize=(10, 8))
        pos_data = pd.Series(morph_analysis['pos_percentages'])
        pos_data = pos_data.sort_values(ascending=False).head(10)  # Take top 10
        
        # POS translation dictionary
        pos_translation = {
            'NOUN': 'Nouns',
            'VERB': 'Verbs',
            'ADJF': 'Full Adj.',
            'ADJS': 'Short Adj.',
            'COMP': 'Comparatives',
            'PRTF': 'Full Participles',
            'PRTS': 'Short Participles',
            'GRND': 'Gerunds',
            'NUMR': 'Numerals',
            'ADVB': 'Adverbs',
            'PREP': 'Prepositions',
            'CONJ': 'Conjunctions',
            'PRCL': 'Particles',
            'INTJ': 'Interjections'
        }
        
        labels = [pos_translation.get(pos, pos) for pos in pos_data.index]
        plt.pie(pos_data.values, labels=labels, autopct='%1.1f%%')
        plt.title('Parts of Speech Distribution')
        pos_path = os.path.join(output_dir, 'pos_distribution.png')
        plt.savefig(pos_path, bbox_inches='tight')
        plt.close()
    
    return {
        'word_lengths': word_lengths_path,
        'pos_distribution': pos_path
    }

def generate_html_report(text, file_name, output_path, vis_paths):
    """Generate HTML report with all metrics"""
    # Extract sentences and words
    se = SentsExtractor()
    sentences = se.extract(text)
    
    we = WordsExtractor(use_lexemes=True, filter_nums=True)
    words = we.extract(text)
    
    # Remove stopwords for diversity analysis
    stopwords = nltk.corpus.stopwords.words('russian')
    content_words = [w for w in words if w not in stopwords and len(w) > 2]
    
    if not content_words:
        raise ValueError("No content words left after processing. Try a longer text.")
    
    # Morphological analysis
    morph_analysis = morphological_analysis(text)
    
    # Syntactic analysis
    syntactic_analysis = syntactic_complexity(text)
    
    # Readability
    readability = readability_index(text)
    
    # Lexical diversity metrics
    tokens = content_words
    metrics = {
        # Basic TTR metrics
        'TTR': ld.ttr(tokens),
        'Root TTR': ld.root_ttr(tokens),
        'Log TTR': ld.log_ttr(tokens),
        'Maas TTR': ld.maas_ttr(tokens),
        
        # Advanced metrics
        'MSTTR (50 words)': ld.msttr(tokens, window_length=50),
        'MATTR (50 words)': ld.mattr(tokens, window_length=50),
        'MATTR (100 words)': ld.mattr(tokens, window_length=100),
        'HD-D': ld.hdd(tokens),
        'MTLD': ld.mtld(tokens),
        'MTLD-MA (Wrap)': ld.mtld_ma_wrap(tokens),
        'MTLD-MA (Bidirectional)': ld.mtld_ma_bid(tokens)
    }
    
    # Basic statistics
    bs = BasicStats(text)
    basic_stats = bs.get_stats()
    
    # POS translation dictionary
    pos_translation = {
        'NOUN': 'Nouns',
        'VERB': 'Verbs',
        'ADJF': 'Full Adj.',
        'ADJS': 'Short Adj.',
        'COMP': 'Comparatives',
        'PRTF': 'Full Participles',
        'PRTS': 'Short Participles',
        'GRND': 'Gerunds',
        'NUMR': 'Numerals',
        'ADVB': 'Adverbs',
        'PREP': 'Prepositions',
        'CONJ': 'Conjunctions',
        'PRCL': 'Particles',
        'INTJ': 'Interjections'
    }
    
    # Generate HTML content
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Linguistic Analysis Report: {os.path.basename(file_name)}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        .container {{
            border: 1px solid #e1e4e8;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            background: #f6f8fa;
        }}
        h1 {{
            color: #0366d6;
            border-bottom: 2px solid #0366d6;
            padding-bottom: 8px;
        }}
        h2 {{
            color: #24292e;
            margin-top: 20px;
        }}
        h3 {{
            color: #24292e;
            margin-top: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        th, td {{
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .chart-container {{
            display: flex;
            justify-content: center;
            margin: 20px 0;
        }}
        .chart-container img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .interpretation {{
            background: #e6f7ff;
            border-radius: 6px;
            padding: 15px;
            margin: 15px 0;
        }}
        .two-column {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }}
        .column {{
            flex: 1;
            min-width: 300px;
            background: #f8f9fa;
            padding: 12px;
            border-radius: 6px;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 15px;
            border-top: 1px solid #eee;
            font-size: 0.9em;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Comprehensive Linguistic Analysis Report</h1>
        <p><strong>File:</strong> {os.path.basename(file_name)}</p>
        <p><strong>Generated:</strong> {os.path.basename(output_path)}</p>
        <p><strong>Date:</strong> {os.popen('date').read().strip()}</p>
        
        <h2>🔤 Basic Statistics (ruTS)</h2>
        <table>
            <tr><th>Statistic</th><th>Value</th></tr>
            <tr><td>Sentences</td><td>{basic_stats['n_sents']}</td></tr>
            <tr><td>Total words</td><td>{basic_stats['n_words']}</td></tr>
            <tr><td>Unique words</td><td>{basic_stats['n_unique_words']}</td></tr>
            <tr><td>Complex words (3+ syllables)</td><td>{basic_stats['n_complex_words']} ({basic_stats['n_complex_words']/basic_stats['n_words']*100:.1f}%)</td></tr>
            <tr><td>Long words (>6 letters)</td><td>{basic_stats['n_long_words']} ({basic_stats['n_long_words']/basic_stats['n_words']*100:.1f}%)</td></tr>
            <tr><td>Polysyllabic words (2+ syllables)</td><td>{basic_stats.get('n_polysyllable_words', 0)} ({basic_stats.get('n_polysyllable_words', 0)/basic_stats['n_words']*100:.1f}%)</td></tr>
            <tr><td>Average word length</td><td>{basic_stats['n_letters']/basic_stats['n_words']:.2f} letters</td></tr>
            <tr><td>Average syllables per word</td><td>{basic_stats['n_syllables']/basic_stats['n_words']:.2f}</td></tr>
        </table>
        
        {f'<div class="chart-container"><img src="{os.path.basename(vis_paths["word_lengths"])}" alt="Word Length Distribution"></div>' if vis_paths["word_lengths"] else ''}
        
        <h2>📈 Lexical Diversity Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th><th>Description</th></tr>
            <tr><td>TTR (Type-Token Ratio)</td><td>{metrics['TTR']:.4f}</td><td>Basic measure of lexical variety</td></tr>
            <tr><td>Root TTR</td><td>{metrics['Root TTR']:.4f}</td><td>Stabilizes dependence on text length</td></tr>
            <tr><td>Log TTR</td><td>{metrics['Log TTR']:.4f}</td><td>Reduces variance for long texts</td></tr>
            <tr><td>Maas TTR</td><td>{metrics['Maas TTR']:.4f}</td><td>Length-invariant for large corpora</td></tr>
            <tr><td>MSTTR (50 words)</td><td>{metrics['MSTTR (50 words)']:.4f}</td><td>Mean TTR in segments of 50 words</td></tr>
            <tr><td>MATTR (50 words)</td><td>{metrics['MATTR (50 words)']:.4f}</td><td>Moving average TTR with 50-word window</td></tr>
            <tr><td>MATTR (100 words)</td><td>{metrics['MATTR (100 words)']:.4f}</td><td>Moving average TTR with 100-word window</td></tr>
            <tr><td>HD-D</td><td>{metrics['HD-D']:.4f}</td><td>Hypergeometric distribution diversity</td></tr>
            <tr><td>MTLD</td><td>{metrics['MTLD']:.4f}</td><td>Mean segment length until TTR=0.72</td></tr>
            <tr><td>MTLD-MA (Wrap)</td><td>{metrics['MTLD-MA (Wrap)']:.4f}</td><td>MTLD with circular window</td></tr>
            <tr><td>MTLD-MA (Bidirectional)</td><td>{metrics['MTLD-MA (Bidirectional)']:.4f}</td><td>MTLD in both directions</td></tr>
        </table>
        
        <h2>🔍 Morphological Analysis (pymorphy2)</h2>
    """
    
    if morph_analysis and morph_analysis.get('pos_percentages'):
        pos_data = morph_analysis['pos_percentages']
        top_pos = sorted(pos_data.items(), key=lambda x: x[1], reverse=True)[:8]
        
        html_content += '<table>\n<tr><th>Part of Speech</th><th>Percentage</th></tr>\n'
        for pos, percent in top_pos:
            translated = pos_translation.get(pos, pos)
            html_content += f'<tr><td>{translated}</td><td>{percent:.1f}%</td></tr>\n'
        html_content += '</table>\n'
        
        # Additional morphological metrics
        html_content += '<div class="two-column">\n'
        
        if morph_analysis['gender_counts']:
            total_gender = sum(morph_analysis['gender_counts'].values())
            if total_gender > 0:
                html_content += '<div class="column">\n<h3>Noun Genders</h3>\n<ul>\n'
                genders = {
                    'masc': 'masculine',
                    'femn': 'feminine',
                    'neut': 'neuter'
                }
                for gender, count in morph_analysis['gender_counts'].items():
                    if gender in genders:
                        html_content += f'<li>{genders[gender]}: {count/total_gender*100:.1f}%</li>\n'
                html_content += '</ul>\n</div>\n'
        
        if morph_analysis['case_counts']:
            total_cases = sum(morph_analysis['case_counts'].values())
            if total_cases > 0:
                html_content += '<div class="column">\n<h3>Noun Cases</h3>\n<ul>\n'
                cases = {
                    'nomn': 'nominative',
                    'gent': 'genitive',
                    'datv': 'dative',
                    'accs': 'accusative',
                    'ablt': 'instrumental',
                    'loct': 'prepositional'
                }
                top_cases = sorted(morph_analysis['case_counts'].items(), key=lambda x: x[1], reverse=True)[:5]
                for case, count in top_cases:
                    if case in cases:
                        html_content += f'<li>{cases[case]}: {count/total_cases*100:.1f}%</li>\n'
                html_content += '</ul>\n</div>\n'
        
        if morph_analysis['tense_counts']:
            total_tense = sum(morph_analysis['tense_counts'].values())
            if total_tense > 0:
                html_content += '<div class="column">\n<h3>Verb Tenses</h3>\n<ul>\n'
                tenses = {
                    'past': 'past',
                    'pres': 'present',
                    'futr': 'future'
                }
                for tense, count in morph_analysis['tense_counts'].items():
                    if tense in tenses:
                        html_content += f'<li>{tenses[tense]}: {count/total_tense*100:.1f}%</li>\n'
                html_content += '</ul>\n</div>\n'
        
        html_content += '</div>\n'
        
        # POS distribution chart
        if vis_paths["pos_distribution"]:
            html_content += f'<div class="chart-container"><img src="{os.path.basename(vis_paths["pos_distribution"])}" alt="POS Distribution" style="max-width: 500px;"></div>\n'
    else:
        html_content += '<p>No morphological analysis available due to insufficient content words.</p>\n'
    
    # Syntactic and readability analysis
    html_content += f"""
        <h2>_Syntax Sentence Structure Analysis</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Average sentence length</td><td>{syntactic_analysis['avg_sentence_length']:.1f} words</td></tr>
            <tr><td>Average clause markers</td><td>{syntactic_analysis['avg_clause_count']:.1f} per sentence</td></tr>
            <tr><td>Readability index</td><td>{readability['readability_score']:.1f}/100 (higher = easier to read)</td></tr>
            <tr><td>Percentage of long words</td><td>{readability['percent_long_words']:.1f}%</td></tr>
        </table>
        
        <div class="interpretation">
            <h2>📖 Interpretation of Key Metrics</h2>
            <p><strong>MATTR & MTLD:</strong> These metrics show lexical richness. High values (>0.6 for MATTR, >60 for MTLD) are typical for literary texts and poetry.</p>
            <p><strong>Readability index:</strong> Values 60-80 correspond to simple language (children's literature, instructions), 40-60 to standard texts (newspapers, magazines), <40 to complex texts (scientific articles).</p>
            <p><strong>Morphology:</strong> High noun percentage (>40%) is typical for scientific style, high verb percentage (>25%) for narrative texts, high adjective percentage (>20%) for descriptive texts.</p>
            <p><strong>Syntax:</strong> Average sentence length >20 words indicates complex syntactic constructions typical of academic texts.</p>
        </div>
        
        <div class="footer">
            <p>Report generated using ruTS, lexical_diversity, NLTK, spaCy and pymorphy2 libraries.</p>
            <p>For more information on metrics, visit: 
                <a href="https://github.com/SergeyShk/ruTS">ruTS GitHub</a> | 
                <a href="https://github.com/kristopherkyle/lexical_diversity">lexical_diversity GitHub</a>
            </p>
        </div>
    </div>
</body>
</html>
"""
    
    # Write HTML to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML report generated: {output_path}")
    if vis_paths["word_lengths"]:
        print(f"📊 Word length distribution chart: {os.path.basename(vis_paths['word_lengths'])}")
    if vis_paths["pos_distribution"]:
        print(f"📊 Parts of speech distribution chart: {os.path.basename(vis_paths['pos_distribution'])}")

def main():
    parser = argparse.ArgumentParser(description='Analyze Russian text for lexical diversity and linguistic features.')
    parser.add_argument('input_file', help='Path to the text file to analyze (.txt)')
    parser.add_argument('-o', '--output', help='Output HTML file name (default: report.html)', default='report.html')
    args = parser.parse_args()

    # Check if input file exists
    if not os.path.isfile(args.input_file):
        print(f"❌ Error: Input file '{args.input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Read input file
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"❌ Error reading file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Clean text
    cleaned_text = clean_text(text)
    word_count = len(cleaned_text.split())
    
    if word_count < 50:
        print(f"⚠️ Warning: Text is short ({word_count} words). Results may not be statistically significant.", file=sys.stderr)
    
    if word_count < 20:
        print(f"❌ Error: Text is too short ({word_count} words). Minimum 20 words required.", file=sys.stderr)
        sys.exit(1)
    
    # Extract words for visualization
    we = WordsExtractor(use_lexemes=True, filter_nums=True)
    words = we.extract(cleaned_text)
    
    # Get directory for output files
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if not output_dir:
        output_dir = os.getcwd()
    
    # Generate visualizations
    print("📊 Generating visualizations...")
    morph_analysis = morphological_analysis(cleaned_text)
    vis_paths = generate_visualizations(cleaned_text, words, morph_analysis, output_dir)
    
    # Generate HTML report
    print(f"📝 Generating HTML report: {args.output}")
    generate_html_report(cleaned_text, args.input_file, args.output, vis_paths)
    
    print(f"\n🎉 Analysis complete!")
    print(f"📂 Input file: {args.input_file}")
    print(f"📄 Report: {args.output}")
    if vis_paths["word_lengths"]:
        print(f"📈 Word lengths chart: {vis_paths['word_lengths']}")
    if vis_paths["pos_distribution"]:
        print(f"📈 POS distribution chart: {vis_paths['pos_distribution']}")

if __name__ == "__main__":
    main()
