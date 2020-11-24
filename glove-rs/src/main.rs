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

fn select_words(tweets: &str, max_word_count: usize) -> Dictionary<'_> {
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

fn construct_cooc(tweets: &str, dict: &Dictionary) -> SparseCooc {
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

        for &w0 in &index_vec {
            for &w1 in &index_vec {
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
    let args: Vec<String> = std::env::args().collect();
    let args = &args[1..];
    if args.len() != 3 {
        panic!("Expected 3 arguments (input, output_words, output_cooc)");
    }

    let tweets = std::fs::read_to_string(&args[0]).expect("Error while reading input file");

    let dict = select_words(&tweets, 10_000);
    let cooc = construct_cooc(&tweets, &dict);

    println!("Cooc size: {}", cooc.counts.len());

    println!("Saving outputs");
    let joined_words = dict.index_to_word.join("\n");
    std::fs::write(&args[1], &joined_words).expect("Error while writing word list");

    let mut array = Array2::default((cooc.counts.len(), 3));
    for i in 0..cooc.counts.len() {
        array[(i, 0)] = cooc.ix[i] as u32;
        array[(i, 1)] = cooc.iy[i] as u32;
        array[(i, 2)] = cooc.counts[i] as u32;
    }

    write_npy(&args[2], &array).expect("Error while writing cooc file");
}
