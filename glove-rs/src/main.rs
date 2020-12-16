use std::cmp::{max, min};
use std::collections::HashMap;

use ndarray::Array2;
use ndarray_npy::write_npy;

struct Dictionary<'s> {
    index_to_word: Vec<&'s str>,
    word_to_index: HashMap<&'s str, usize>,
}

impl<'s> Dictionary<'s> {
    fn len(&self) -> usize {
        self.index_to_word.len()
    }
}

#[derive(Default)]
struct SparseCooc {
    ix: Vec<usize>,
    iy: Vec<usize>,
    counts: Vec<usize>,
}

fn select_words(tweets: &str, max_word_count: usize, punctuation: bool) -> Dictionary<'_> {
    println!("Counting words");
    let mut word_counts = HashMap::new();

    for word in tweets.split_ascii_whitespace() {
        *word_counts.entry(word).or_insert(0) += 1;
    }

    println!("Picking words");
    let mut word_count_list: Vec<_> = word_counts.iter()
        .map(|(s, c)| (*s, *c))
        .collect();
    word_count_list.sort_by_key(|&(_, c)| std::cmp::Reverse(c));

    //TODO maybe remove () as well because it's kind of cheating?

    let index_to_word: Vec<&str> = word_count_list.iter()
        .map(|&(w, _)| w)
        .filter(|w| w.chars().any(char::is_alphanumeric))
        .take(max_word_count)
        .collect();

    let word_to_index: HashMap<&str, usize> = index_to_word.iter()
        .enumerate()
        .map(|(i, w)| (*w, i))
        .collect();

    Dictionary {
        index_to_word,
        word_to_index,
    }
}

fn construct_cooc(tweets: &str, dict: &Dictionary, context_dist: Option<usize>) -> SparseCooc {
    let tweet_count = tweets.lines().count();

    println!("Constructing cooc");
    let mut dense_cooc = vec![0; dict.len() * dict.len()];

    let mut index_vec: Vec<usize> = Vec::new();

    for (i, tweet) in tweets.lines().enumerate() {
        if i % (tweet_count / 10) == 0 {
            println!("progress {}", i as f32 / tweet_count as f32);
        }

        index_vec.clear();
        index_vec.extend(tweet.split(' ').filter_map(|w| dict.word_to_index.get(w)));

        for (i0, &w0) in index_vec.iter().enumerate() {
            let context = match context_dist {
                None => 0..index_vec.len(),
                Some(context_dist) => max(0, i0 - context_dist)..min(index_vec.len(), i0 + context_dist)
            };

            for i1 in context {
                let w1 = index_vec[i1];
                dense_cooc[w0 + w1 * dict.len()] += 1;
            }
        }
    }

    println!("Converting to sparse");
    let mut sparse_cooc = SparseCooc::default();

    for w0 in 0..dict.len() {
        for w1 in 0..dict.len() {
            let count = dense_cooc[w0 + w1 * dict.len()];
            if count != 0 {
                sparse_cooc.ix.push(w0);
                sparse_cooc.iy.push(w1);
                sparse_cooc.counts.push(count)
            }
        }
    }

    sparse_cooc
}

fn main() {
    //arg handling
    let args: Vec<String> = std::env::args().collect();
    let args = &args[1..];
    if args.len() != 6 {
        panic!("Expected 6 arguments (input, word_count, context_dist, punctuation, output_words, output_cooc)");
    }

    let word_count: usize = args[1].parse().expect("word_count must be an integer");
    let context_dist: usize = args[2].parse().expect("context_dist must be an integer");
    let context_dist = if context_dist == 0 { None } else { Some(context_dist) };
    let punctuation: bool = args[3].parse().expect("punctuation must be a bool");

    let input_path = &args[0];
    let output_words_path = &args[4];
    let output_cooc_path = &args[5];

    //start doing useful stuff
    let tweets = std::fs::read_to_string(input_path).expect("Error while reading input file");

    let dict = select_words(&tweets, word_count, punctuation);
    let cooc = construct_cooc(&tweets, &dict, context_dist);

    println!("Cooc size: {}", cooc.counts.len());

    println!("Saving outputs");
    let joined_words = dict.index_to_word.join("\n");
    std::fs::write(output_words_path, &joined_words).expect("Error while writing word list");

    let mut array = Array2::default((cooc.counts.len(), 3));
    for i in 0..cooc.counts.len() {
        array[(i, 0)] = cooc.ix[i] as u32;
        array[(i, 1)] = cooc.iy[i] as u32;
        array[(i, 2)] = cooc.counts[i] as u32;
    }

    write_npy(output_cooc_path, &array).expect("Error while writing cooc file");
}
