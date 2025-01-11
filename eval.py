"""Command-line script for standalone ICDAR map-text competition evaluation

Copyright 2024, 2025 Jerod Weinman

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>."""

import json
import argparse
import logging
import re
import statistics

from functools import reduce, partial
from multiprocessing import Pool
from typing import Union, Tuple, Any, Optional, Callable

import scipy  # type: ignore
import numpy as np
import numpy.typing as npt
import shapely  # type: ignore

from pyeditdistance.distance import normalized_levenshtein  # type: ignore

# Minimum area of a polygon to be considered for area-based processing
# Values below this will be treated as an IoU or overlap percentage of zero.
POLY_EPSILON = 1e-5

# Bipartite match score value for a correspondence with an "ignore" ground truth
IGNORE_EPSILON = 1e-12

parser = argparse.ArgumentParser(
    description='Map Text Competition Task Evaluation')
parser.add_argument('--gt', type=str, required=True,
                    help="Path to the ground truth JSON file")
parser.add_argument('--pred', type=str, required=True,
                    help="Path to the predictions JSON file")
parser.add_argument('--output', type=str, required=False, default=None,
                    help="Path to the JSON file containing results")
parser.add_argument('--task', type=str, required=True,
                    choices=['det', 'detedges',
                             'detrec' ,'detrecedges'],
                    help="Task to evaluate against")
parser.add_argument('--iou-threshold', type=float, default=0.5,
                    help="Minimum IoU for elements to be considered a match")
parser.add_argument('--use-tightness', default=True,
                    action=argparse.BooleanOptionalAction,
                    help="Include tightness in evaluation")
parser.add_argument('--parallel', type=str, required=False, default='none',
                    choices=['none', 'spark', 'pool'],
                    help="Process evaluation in parallel using multiprocessing or Apache Spark")
parser.add_argument('--gt-regex', type=str, required=False,
                    help="Regular expression to filter image keys for evaluation")

# Type aliases for hints
# NB: "type" omitted for compatibility with Python3.9, used by RRC platform
WordData = dict
GroupData = dict  # list[WordData]
ImageData = list[GroupData]
Number = Union[int,float]


def warn_image_keys( gt_keys: set,
                     preds_keys: set ):
    """Log warnings about image key discrepancies between ground truth and
    prediction files (i.e., predictions missing an image from ground truth)."""

    for gt in gt_keys:
        if gt not in preds_keys:
            logging.warning('Key %s missing from predictions', gt)


def add_word_polygons( data: dict[str,ImageData] ):
    """Add the field 'geometry', a shapely.geometry.Polygon, to each word.
    It is constructed from the 'vertices' of the word and closed by repeating
    the first vertex.

    Returns
      Nothing; called for side-effect (mutating data)
    """
    for image_groups in data.values():
        for group in image_groups:
            for word in group['words']:
                points = word['vertices']
                points_closed = points + [points[0]]  # Close by repeating first
                word['geometry'] = shapely.geometry.Polygon(points_closed)


def verify_predictions_format( preds: list,
                               is_e2e: bool = True ) -> bool:
    """ Verify predictions format, read from JSON file.

    Arguments
      preds : Loaded predictions JSON
      is_e2e  : Whether to check for mandatory additional fields (default=True)
    Returns
      True
    Raises
      TypeErrors or ValueError if the parse fails type or value checks."""

    IMAGE_KEY_TYPES = {'image': str,
                       'groups': list }
    WORD_KEY_TYPES: dict[str,Any] = { 'vertices': list }

    if is_e2e:  # Require text
        WORD_KEY_TYPES['text'] = str

    WORD_KEY_SET = set(WORD_KEY_TYPES.keys())

    if not isinstance(preds, list):  # Check top level is a list
        raise TypeError('Expected predictions top-level to be a list (of images), found {}'.format(type(preds)))

    for entry in preds:  # Check each list item (i.e., image)

        if not isinstance(entry, dict):  # Check each image is a dict
            raise TypeError('Expected predictions entry for each image to be a dict, found {}'.format(type(entry)))

        if entry.keys() != IMAGE_KEY_TYPES.keys():  # Check image dict keyset
            raise ValueError('Expected predictions entry for image to contain keyset {}, found {}'.format(IMAGE_KEY_TYPES.keys(),entry.keys()))

        for (k,v) in entry.items():  # Check image dict value types
            if not isinstance(v, IMAGE_KEY_TYPES[k]):
                raise TypeError('Expected predictions entry for image key {} to have type {}, found {}'.format(k,IMAGE_KEY_TYPES[k],type(v)))

        # Image keys all check out. Time to descend to check their values.
        for group in entry['groups']:

            if not isinstance(group, list):  # Check each group is a list (of words)
                raise TypeError('Expected predictions entry for group to have type list (of words), found {}'.format(type(group)))

            for word in group: # Check each word
                if not isinstance(word, dict):  # Check each word is a dict
                    raise TypeError('Expected predictions entry for word to have type dict, found {}'.format(type(word)))

                # NB: Check for subset; ignore extra keys
                if not WORD_KEY_SET.issubset(word.keys()):  # Check word dict keyset
                    raise ValueError('Expected predictions entry for word to contain keyset {}, found {}'.format(WORD_KEY_TYPES.keys(),word.keys()))

                for (k,v) in word.items():  # Check image dict value types
                    if k not in WORD_KEY_TYPES:  # Superfluous key; ignore
                        continue

                    if not isinstance(v, WORD_KEY_TYPES[k]):
                        raise TypeError('Expected predictions for image key {} to have type {}, found {}'.format(k,WORD_KEY_TYPES[k],type(v)))

                # Specifically check vertices
                if len(word['vertices']) < 3:
                    raise ValueError('Expected at least three vertices for word, found {}'.format(word['vertices']))

                for vertex in word['vertices']:
                    if not isinstance(vertex, list) or \
                      len(vertex) != 2 or \
                      not all( isinstance( c, (float,int)) for c in vertex):
                        raise TypeError('Expected vertex to be a list of two numbers, found {}'.format(vertex))
    return True


def verify_ground_truth_format( gt_raw: list ):
    """Verify ground truth format read from JSON file. Returns nothing,
        but raises errors if the parse fails type or value checks.
    """

    IMAGE_KEY_TYPES = {'image' : str,
                       'groups': list }
    WORD_KEY_TYPES = { 'vertices' : list,
                       'text'     : str,
                       'illegible': bool,
                       'truncated': bool }
    WORD_KEY_SET = set(WORD_KEY_TYPES.keys())

    if not isinstance(gt_raw,list):  # Check top level is a list
        raise TypeError(
            'Expected ground truth top-level to be a list (of images), found {}'.format(type(gt_raw)))

    for entry in gt_raw:  # Check each list item (i.e., image)

        if not isinstance(entry, dict):  # Check each image is a dict
            raise TypeError('Expected ground truth entry for each image to be a dict, found {}'.format(type(entry)))

        if entry.keys() != IMAGE_KEY_TYPES.keys():  # Check image dict keyset
            raise ValueError('Expected ground truth entry for image to contain keyset {}, found {}'.format(IMAGE_KEY_TYPES.keys(),entry.keys()))

        for (k,v) in entry.items():  # Check image dict value types
            if not isinstance(v, IMAGE_KEY_TYPES[k]):
                raise TypeError('Expected ground truth entry for image key {} to have type {}, found {}'.format(k,IMAGE_KEY_TYPES[k],type(v)))

        # Image keys all check out. Time to descend to check their values.
        for group in entry['groups']:

            if not isinstance(group,list):  # Check each group is a list (of words)
                raise TypeError('Expected ground truth entry for group to have type list (of words), found {}'.format(type(group)))

            for word in group:  # Check each word
                if not isinstance(word, dict):  # Check each word is a dict
                    raise TypeError('Expected ground truth entry for word to have type dict, found {}'.format(type(word)))

                # NB: Check for subset; ignore extra (unexpected) keys in ground truth
                if not WORD_KEY_SET.issubset(word.keys()):  # Check word dict keyset
                    raise ValueError('Expected ground truth entry for word to contain keyset {}, found {}'.format(WORD_KEY_TYPES.keys(),word.keys()))

                for (k,v) in word.items():  # Check word dict value types
                    if k not in WORD_KEY_TYPES:  # Superfluous key; ignore
                        # Issue warning? Could get clogged up.
                        continue
                    if not isinstance(v, WORD_KEY_TYPES[k]):
                        raise TypeError('Expected ground truth entry for word key {} to have type {}, found {}'.format(k,WORD_KEY_TYPES[k],type(v)))

                # Specifically check vertices
                if len(word['vertices']) < 3:
                    raise ValueError('Expected at least three vertices for word, found {}'.format(word['vertices']))

                for vertex in word['vertices']:
                    if not isinstance(vertex, list) or \
                      len(vertex) != 2 or \
                      not all( isinstance( c, (float,int)) for c in vertex):
                        raise TypeError('Expected vertex to be a list of two numbers, found {}'.format(vertex))


def load_ground_truth( gt_file: str,
                       is_e2e: bool,
                       image_regex: Optional[str] = None,
                       verify: bool = True ) -> dict[str,ImageData]:
    """ Load the ground truth file and optionally verify contents format

    Arguments
      gt_file : Path to the ground truth JSON file (see competition format)
      is_e2e : Whether the evaluation is end-to-end (i.e., includes recognition)
      image_regex : Regular expression to filter image keys (default=None)
      verify : Whether to verify the ground truth file (default=True)
    Returns
      gt_anno : Dict indexed by the image id, giving the list of groups
    """

    # NB: This is the one place, aside from the format parser where the ignore
    #  fields require processing.
    # TODO(jjw): Generalize to any() over defined list of named ignore fields?
    def fold_word_ignores(group: list[WordData]) -> list[WordData]:
        """Combine cases causing a ground truth to be ignored into a single
        ignore flag field"""
        if is_e2e:
            return [ {'text': word['text'],
                      'vertices': word['vertices'],
                      'ignore':  word['truncated'] or word['illegible'] }
                     for word in group ]
        else:
            return [ {'vertices': word['vertices'],
                      'ignore':  word['truncated'] or word['illegible'] }
                     for word in group ]

    logging.info('Loading ground truth annotations...')

    with open(gt_file, encoding='utf-8') as fd:
        gt_raw = json.load(fd)

    if verify:
        verify_ground_truth_format(gt_raw)

    # Re-index: image-->groups, wrapping group lists to dicts (for more fields)
    gt = {}
    regex = re.compile(image_regex) if image_regex else None
    for entry in gt_raw:
        if regex and not regex.match(entry['image']):  # Skip if no match
            continue
        groups = entry['groups']
        groups = [ { 'words' : fold_word_ignores(group) } for group in groups ]
        gt[entry['image']] = groups

    if len(gt) == 0:
        raise ValueError('No ground truth images for evaluation')

    # Add additional/transformed fields for evaluation processing
    add_word_polygons(gt)

    return gt


def load_predictions( preds_file: str,
                      is_e2e: bool,
                      image_regex: Optional[str] = None ) \
                      -> dict[str,ImageData]:
    """ Load the predictions file and verify contents format

    Arguments
      preds_file : Path to the predictions JSON file (see competition format)
      is_e2e: Whether the evaluation is end-to-end (i.e., includes recognition)
    Returns
      preds : Dict indexed by the image id, giving the list of groups
    """

    logging.info('Loading predictions...')

    with open(preds_file, encoding='utf-8') as fd:
        preds_raw = json.load(fd)

    verify_predictions_format(preds_raw, is_e2e)

    # Re-index: image-->groups, wrapping group lists to dicts (for more fields)
    preds = {}
    regex = re.compile(image_regex) if image_regex else None
    for entry in preds_raw:
        if regex and not regex.match(entry['image']):  # Skip if no match
            continue
        groups = [ { 'words' : group } for group in entry['groups'] ]
        preds[entry['image']] = groups

    # Add additional/transformed fields for evaluation processing
    add_word_polygons(preds)

    return preds


def get_adjacency_matrix( groups: list[GroupData] ) -> npt.NDArray[np.bool_]:
    """Return a directed adjacency matrix for the flat/linear words in the groups

    Arguments
      groups: List of dicts containing ground truth or predicted elements
    Returns
      edges: NxN numpy bool array where N is the total number of words in groups
               and edges[i,j]=1 iff words i,j link in that order
    """
    
    index = 0 # current word flat/linear index

    num_words = sum( [len(group['words']) for group in groups] )

    edges = np.zeros( (num_words, num_words), dtype=np.bool_ )
    
    for group in groups:
        for i, word in enumerate(group['words']):
            if i != 0: # not first word in group?
                edges[index-1,index] = 1
            index = index + 1
    return edges


def get_eval_adjacency_matrices( gt_edges: npt.NDArray[np.bool_],
                                  pred_edges: npt.NDArray[np.bool_],
                                  matches_gt: npt.NDArray[np.uint],
                                  matches_pred: npt.NDArray[np.uint]) :
    """Get adjacency matrices re-ordered for evaluation so that the
    indices align for the matched items. Index ordering is: matches,
    unmatched ground truths, unmatched predictions.

    Parameters
      gt_edges:     MxM numpy bool array of ground truth word links
      pred_edges:   NxN numpy bool array of predicted word links
      matches_gt:   Length T numpy array of values in [0,M) indicating ground
                      truth element matched (corresponds to entries in
                      matches_pred)
      matches_pred: Length T numpy array of values in [0,N) indicating
                      predicted element matched (corresponds to entries in
                      matches_gt)
    Returns
      gt_edges_eval:   PxP numpy bool area of re-indexed ground truth word links
      pred_edges_eval: PxP numpy bool area of re-indexed predicted word links
      gt_inverted:     Length M numpy array of unique values in [0,P); re-indexing
      pred_inverted:   Length N numpy array of unique values in [0,P); re-indexing
    """

    def indices_complement(indices, size):
        """Generate the complementary set of linear indices"""
        logical = np.ones( size, dtype=np.bool_)
        logical[indices] = 0
        return np.nonzero(logical)[0] # Convert from logical to indices

    def inverted_indices(matches, offset, nonmatches):
        """Invert index from original domain (1:N) to evaluation domain (1:M)"""
        inverted = np.zeros( len(matches) + len(nonmatches), dtype=np.uint )
        # Matches go at the "front" of the evaluation domain
        inverted[matches] = list(range(len(matches)))
        # Non-matches go at the end, after the matches and perhaps after an offset
        inverted[nonmatches] = [len(matches) + offset + n
                                    for n in range(len(nonmatches))]
        return inverted
        
    def rebuild_edges( edges, matches, offset, nonmatches, size):
        """Re-indexes to build adjacency matrix"""
        new_edges = np.zeros( (size,size), np.bool_)
        inverted = inverted_indices(matches, offset, nonmatches)
        edges_i,edges_j = np.nonzero(edges) # Indices of original edges
        new_edges[ inverted[edges_i], inverted[edges_j]] = 1
        return new_edges, inverted
            
    # Get arrays of unmatched items' indices
    unmatched_gt = indices_complement( matches_gt, np.shape(gt_edges)[0] )
    unmatched_pred = indices_complement( matches_pred, np.shape(pred_edges)[0] )

    # Size of new adjacency matrices
    assert( len(matches_gt) == len(matches_pred) )
    sz = len(matches_gt) + len(unmatched_gt) + len(unmatched_pred)

    # Construct the aligned adjacency matrices for ground truth and predictions
    gt_edges_eval, gt_inverted = rebuild_edges(
        gt_edges, matches_gt, 0, unmatched_gt, sz)
    pred_edges_eval, pred_inverted = rebuild_edges(
        pred_edges, matches_pred, len(unmatched_gt), unmatched_pred, sz)
    
    return gt_edges_eval, pred_edges_eval, gt_inverted, pred_inverted


def evaluate_edges(gt_ignore: npt.NDArray[np.bool_],
                   gt_edges: npt.NDArray[np.bool_],
                   pred_edges: npt.NDArray[np.bool_],
                   matches_gt: npt.NDArray[np.uint],
                   matches_pred: npt.NDArray[np.uint]):
    """Evaluate edge link predictions

    Parameters
      gt_ignore:    Numpy array of values in [0,M) indicating ground truth 
                      elements that are to be ignored
      gt_edges:     MxM numpy bool array where M is the number of ground truth 
                       words and edges[i,j]=1 iff words i,j link in that order
      pred_edges:   NxN numpy bool array where N is the number of predicted
                       words and edges[i,j]=1 iff words i,j link in that order
      matches_gt:   Length T numpy array of values in [0,M) indicating ground
                      truth element matched (corresponds to entries in
                      matches_pred)
      matches_pred: Length T numpy array of values in [0,N) indicating
                      predicted element matched (corresponds to entries in
                      matches_gt)
     Returns
       results: Dict with keys 'edges_tp', 'edges_gt', and 'edges_pred'
                  representing the counts of true positives, non-ignored
                  ground truth edges, and non-ignored predicted edges, resp.
       stats:   Dict of summary statistics with keys 'edge_recall',
                  'edge_precision', and 'edge_fscore'

     """

    # Re-process adjacency matrices to align indices for evaluation
    gt_edges_eval, pred_edges_eval, gt_inverted, pred_inverted = \
      get_eval_adjacency_matrices( gt_edges, pred_edges,
                                   matches_gt, matches_pred )

    # Find rows/columns to ignore in evaluation because they correspond to
    # ground truth ignore items
    eval_ignore = gt_inverted[gt_ignore]
    eval_count = np.ones( np.shape(gt_edges_eval)[0], dtype=np.bool_ )
    eval_count[eval_ignore] = False

    gt_eval = gt_edges_eval[eval_count,:]
    gt_eval = gt_eval[:,eval_count]

    pred_eval = pred_edges_eval[eval_count,:]
    pred_eval = pred_eval[:,eval_count]

    # Tabulate results
    edges_tp = np.sum( np.logical_and( gt_eval, pred_eval) )
    total_gt = np.sum( gt_eval )
    total_pred = np.sum( pred_eval)

    results = { 'edges_tp'   : int(edges_tp),
                'edges_gt'   : int(total_gt),
                'edges_pred' : int(total_pred) }
        
    stats = get_stats( edges_tp, total_gt, total_pred, prefix='edges_')
    
    return results, stats
    

def calc_score_pairs( gt: list[WordData],
                      pred: list[WordData],
                      can_match: Callable[[WordData,WordData,float],
                                          bool],
                      score_match: Callable[[WordData,WordData,float],
                                            bool] ) \
                      -> Tuple[npt.NDArray[np.bool_],
                               npt.NDArray[np.double],
                               npt.NDArray[np.double]]:
    """Return the correspondence score and IoU between all pairs of shapes.

    Arguments
      gt :  List of dicts containing ground truth elements (each has the field
           'geometry' among others).
      pred : List of dicts containing predicted elements (each has the field
             'geometry' among others).
      can_match: Predicate indicating whether ground truth and prediction are
                   valid correspondence candidates
      score_match: Function taking ground truth and predicted word dicts with
                    their pre-calculated iou score and returning their match
                    score (assumes they are valid matches)
    Returns
      allowed: MxN numpy bool array of can_match(g,d) correspondence candidates
      scores : MxN numpy float array of compatibility scores
      ious : MxN numpy float array of IoU values

      where M is len(gt) and N is len(pred).
    """
    def calc_iou( p, q ) -> float :
        """ Return the IoU between two shapes """
        if p.intersects(q) and \
          p.area >= POLY_EPSILON and q.area >= POLY_EPSILON:
            intersection = p.intersection(q).area
            union = p.union(q).area
            return intersection / (union + POLY_EPSILON)
        else:
            return 0.0

    allowed = np.zeros( (len(gt),len(pred)), dtype=np.bool_ )
    scores = -np.ones( (len(gt),len(pred)), dtype=np.double )
    ious = np.zeros( (len(gt),len(pred)), dtype=np.double )

    for i,gt_el in enumerate(gt):
        for j,pred_el in enumerate(pred):
            try:
                the_iou = calc_iou( gt_el['geometry'], pred_el['geometry'])
            except shapely.errors.GEOSException as e:
                logging.warning('Error at iou(%d,%d): %s}. Skipping ...',i,j,e)
                continue

            if the_iou != 0:
                ious[i,j] = the_iou

            allowed[i,j] = can_match( gt_el, pred_el, the_iou)

            if allowed[i,j]:
                scores[i,j] = score_match( gt_el, pred_el, the_iou)

    return allowed,scores,ious


def get_stats( num_tp: Number,
               num_gt: Number,
               num_pred: Number,
               tot_iou: Optional[Number] = None,
               prefix: str = '') -> dict[str,float] :
    """Calculate statistics: recall, precision, fscore, tightness, quality, and
    hmean from accumulated totals.

    Arguments
      num_tp:   Number of true positives
      num_gt:   Number of ground truth positives in the evaluation
      num_pred: Number of predicted positives in the evaluation
      tot_iou:  Total IoU scores among true positives (default=None)
      prefix:   Optional prefix for return result keys (default='')
    Returns
      dict containing statistics with keys 'recall', 'precision', 'fscore',
        'tightness' (average IoU score), 'quality' (product of fscore and
        tightness), and 'hmean' (harmonic mean of recall, precision, and
        tightness).
    """
    recall    = float(num_tp) / num_gt   if (num_gt > 0)   else 0.0
    precision = float(num_tp) / num_pred if (num_pred > 0) else 0.0
    fscore    = 2.0*recall*precision / (recall+precision) \
        if (recall + precision > 0) else 0.0


    stats = {prefix+'recall'    : recall,
             prefix+'precision' : precision,
             prefix+'fscore'    : fscore}

    if tot_iou is not None:
        tightness = tot_iou / float(num_tp)  if (num_tp > 0)   else 0.0
        quality = tightness * fscore
        hmean = statistics.harmonic_mean([recall,precision,tightness])
        stats[prefix+'tightness'] = tightness
        stats[prefix+'quality']   = quality
        stats[prefix+'hmean']     = hmean
        
    return stats


def get_final_stats(totals: dict[str,Number],
                    task: str,
                    use_tightness: bool) -> dict[str,Number] :
    """Process totals to produce final statistics for the entire data set.

    Arguments
      totals : Dict with keys 'tp', 'total_gt', 'total_pred',
                 'total_tightness', and (if 'rec' in task), 'total_rec_score'.
      task : String containing a valid task (cf parser)
      use_tightness : Include tightness in the final harmonic mean
    Returns
      dict containing statistics with keys 'recall', 'precision',
        'fscore', 'tightness' (average IoU score),  'quality'
        (product of fscore and tightness);  (if 'rec' in task)
        'char_accuracy' and 'char_quality' (product of det_quality and
        char_accuracy); (if 'edges' in task) 'edges_recall', 'edges_precision';
        and 'hmean', the harmonic mean of recall, precision, (if use_tightness)
        tightness, (if 'edges in task) edges_recall, edges_precision, and
        (if 'rec' in task) char_accuracy.
    """
    final_stats = get_stats( totals['tp'],
                             totals['total_gt'],
                             totals['total_pred'],
                             totals['total_tightness'])

    # Terms for the final harmonic mean (may be appended)
    quantities = [ final_stats['recall'], final_stats['precision'] ]

    if use_tightness:
        quantities.append(final_stats['tightness']) # Include tightness in Hmean

    if 'rec' in task:
        if totals['tp'] > 0:
            accuracy = totals['total_rec_score'] / float(totals['tp'])
        else:
            accuracy = 0.0
        final_stats['char_accuracy'] = accuracy
        final_stats['char_quality'] = accuracy * final_stats['quality']

        quantities.append(accuracy) # Include character accuracy in Hmean

    if 'edges' in task:
        edges_stats = get_stats( totals['edges_tp'],
                                 totals['edges_gt'],
                                 totals['edges_pred'],
                                 prefix='edges_' )
        final_stats = final_stats | edges_stats

        # Include edge stats in Hmean
        quantities += [ edges_stats['edges_recall'],
                        edges_stats['edges_precision'] ]

    final_stats['hmean'] = statistics.harmonic_mean(quantities)
        
    return final_stats


def find_matches(allowable: npt.NDArray[np.bool_],
                 scores: npt.NDArray[np.double],
                 ious: npt.NDArray[np.double] ) \
                 -> Tuple[npt.NDArray[np.uint],
                          npt.NDArray[np.uint],
                          npt.NDArray[np.double]]:
    """Optimize the bipartite matches and filter them to allowable matches.
    Parameters
      allowable:      MxN numpy bool array of valid correspondence candidates
      scores:         MxN numpy float array of match candidate scores
      ious:           MxN numpy float array of IoU scores
    Returns
      matches_gt:   Length T numpy array of values in [0,M) indicating ground
                      truth element matched (corresponds to entries in
                      matches_pred)
      matches_pred: Length T numpy array of values in [0,N) indicating
                      predicted element matched (corresponds to entries in
                      matches_gt)
      matches_ious: Length T numpy array of matches' values from ious
    """
    matches_gt,matches_pred = \
        scipy.optimize.linear_sum_assignment(scores, maximize=True)

    # A maximal bipartite matching, which scipy linear sum assignment algorithm
    # appears to give, may include non-allowable matchings due to lack of
    # alternatives. Therefore, these must be removed from the final list.
    # (This is likely more straightforward than fiddling with returned indices
    # after pre-filtering rows/columns that have no viable partners).
    matches_valid = allowable[matches_gt,matches_pred]
    matches_gt    = matches_gt[matches_valid]
    matches_pred  = matches_pred[matches_valid]

    matches_ious  = ious[matches_gt,matches_pred]

    return matches_gt, matches_pred, matches_ious


def evaluate_image( gt: list[GroupData],
                    pred: list[GroupData],
                    task: str,
                    can_match: Callable[[WordData,WordData,float],
                                        bool],
                    score_match: Callable[[WordData,WordData,float],
                                          bool] ) \
                    -> Tuple[dict[str,Number], dict[str,Number]]:
    """Apply the appropriate evaluation scheme to lists of ground truth and
    prediction elements from the same image.

    Arguments
      gt: List of dicts containing ground truth elements (each has the fields
           'geometry', 'text', and 'ignore').
      pred: List of dicts containing predicted elements for evaluation (each
             has the fields 'geometry' and (if task contains 'rec') 'text'.
      task: string describing the task (det, detlink, detedge, detrec, 
             detreclink,  detrecedge)
      can_match: Predicate indicating whether ground truth and prediction are
                   valid correspondence candidates
      score_match: Function taking ground truth and predicted word dicts with
                    their pre-calculated iou score and returning their match
                    score (assumes they are valid matches)
    Returns
      results : dict containing totals for the accumulator
      stats : dict containing statistics for this image
    """

    if 'edges' in task: # Construct adjacency matrices before flattening
        gt_edges = get_adjacency_matrix(gt)
        pred_edges = get_adjacency_matrix(pred)

    # Flatten from list of groups to list of words for evaluation
    gt = [ word for group in gt
               for word in group['words'] ]
    pred = [ word for group in pred
                 for word in group['words'] ]

    allowed, scores, ious = calc_score_pairs( gt, pred, can_match, score_match )
    matches_gt, matches_pred, matches_ious = find_matches(allowed, scores, ious)

    # Mark as ignorable any predicted regions that matched an ignored region
    matches_ignore = np.asarray([gt[i]['ignore'] for i in matches_gt])
    matches_count  = np.logical_not(matches_ignore)
    num_matches_ignore = np.sum(matches_ignore)

    # Count the total number of ground truth entries marked as ignore
    num_gt_ignore  = len( [ el for el in gt if el['ignore'] ] )

    total_pred = len(pred)  - num_matches_ignore
    total_gt   = len(gt)    - num_gt_ignore

    # Discount predictions that matched to an ignore
    num_tp = len(matches_pred) - num_matches_ignore

    # Accumulate tightness for matches that count (not ignorable)
    total_tightness = np.sum(matches_ious[matches_count])

    results = { 'tp' : int(num_tp),
                'total_gt' : int(total_gt),
                'total_pred' :  int(total_pred),
                'total_tightness' : total_tightness }

    stats = get_stats( num_tp, total_gt, total_pred, total_tightness )

    if 'edges' in task:
        gt_ignore = np.asarray( [ g['ignore'] for g in gt], dtype=np.bool_)
        edge_results, edge_stats = evaluate_edges( gt_ignore,
                                                   gt_edges, pred_edges,
                                                   matches_gt, matches_pred)
        results = results | edge_results
        stats   = stats   | edge_stats
        
    if 'rec' in task:
        # measure text (mis)prediction true positives
        text_score_matches = [ str_score( gt[g]['text'], pred[p]['text'] )
          for (g,p) in zip(matches_gt[matches_count],
                           matches_pred[matches_count]) ]
        # tally scores among true positives
        total_rec_score = sum( text_score_matches )

        accuracy = total_rec_score / float(num_tp) if (num_tp > 0) else 0.0

        stats['char_accuracy'] = accuracy
        stats['char_quality']  = accuracy * stats['quality']

        results['total_rec_score'] = total_rec_score

    return results, stats


def evaluate(gt: dict[str,ImageData],
             pred: dict[str,ImageData],
             task: str,
             can_match: Callable[[WordData,WordData,float],
                                 bool],
             score_match: Callable[[WordData,WordData,float],
                                   bool],
             use_tightness: bool ) \
             -> Tuple[dict[str,float], dict[str,dict[str,float]]]:
    """Run the primary evaluation protocol over all images

    Returns:
      final_stats : dict containing pooled statistics for the entire data set
      stats : dict containing statistics for each image in the data set
    """
    def accumulate( totals: dict[str,float], results: dict[str,float] ):
        """Side-effect totals by accumulating matching keys of results"""
        for (k,v) in results.items():
            totals[k] += v

    # initialize accumulator
    totals = { 'tp' : 0,
               'total_gt' : 0,
               'total_pred' : 0,
               'total_tightness' : 0.0 }
    if 'rec' in task:
        totals['total_rec_score'] = 0.0
    if 'edges' in task:
        totals = totals | { 'edges_tp':0, 'edges_gt':0, 'edges_pred':0 }
    stats = {}  # Collected per-image statistics

    for (img,gt_groups) in gt.items():  # Process each image
        pred_groups = pred[img] if img in pred else []

        img_results, img_stats = evaluate_image( gt_groups, pred_groups,
                                                 task,
                                                 can_match, score_match )

        accumulate(totals,img_results)
        stats[img] = img_stats

    final_stats = get_final_stats( totals, task, use_tightness )  # Process totals

    #print(totals)
    return final_stats, stats


def sum_reduce_dict( a: dict[str,float], b: dict[str,float] ) -> dict[str,float]:
    """Reduce dictionaries by summing matching keys"""
    return { k : v + b[k] for (k,v) in a.items() }


def flatten_zip_dict( gt: dict[str,ImageData],
                      pred: dict[str,ImageData],
                      task: str) \
                      -> Tuple[list[str],list[Tuple[ImageData,ImageData]]]:
    """Helper for parallel processing functions. Zips corresponding ground truth
    and prediction entries into list of tuples, according to the task."""
    img_keys = list(gt.keys())  # cache keys to be certain of fixed ordering

    # Reformulate gt,pred dicts as list of tuples (for parallelized data frame)
    data = [ (gt[img], pred[img] if img in pred else [])
             for img in img_keys ]

    # flatten entries to lists of words for evaluation
    data = [ ( [word for group in g for word in group['words'] ],
               [word for group in p for word in group['words'] ])
                 for (g,p) in data ]
    return img_keys,data


def spark_evaluate(gt: dict[str,ImageData],
                   pred: dict[str,ImageData],
                   task: str,
                   can_match: Callable[[WordData,WordData,float],
                                       bool],
                   score_match: Callable[[WordData,WordData,float],
                                         bool],
                   use_tightness: bool,
                   images_per_slice: int = 10 ) -> \
                   Tuple[dict[str,float], dict[str,dict[str,float]]]:
    """Run the primary evaluation protocol in parallel using Apache Spark

    Returns:
      final_stats : dict containing pooled statistics for the entire data set
      stats : dict containing statistics for each image in the data set
    """

    img_keys, data = flatten_zip_dict(gt,pred,task) # zip image keys for map

    # local import so script can run in sequential mode without spark installed
    from pyspark.sql import SparkSession  # pylint: disable=import-outside-toplevel

    spark_session = SparkSession.builder.appName("MapTextEval").getOrCreate()
    spark_session.sparkContext.addPyFile(__file__)  # Ensure serializability

    num_slices = len(data) // images_per_slice
    data_rdd = spark_session.sparkContext.parallelize(data,numSlices=num_slices)
    # Parallel run: produces list of tuples: [(totals, image_stats), ...]
    results_rdd = data_rdd.map( lambda gp: evaluate_image(gp[0], gp[1],
                                                          task,
                                                          can_match,
                                                          score_match) )
    results_rdd.persist()  # Cache results to avoid subsequent re-computation
    # Splice totals and reduce by summing, then splice per-image stats
    totals = results_rdd.map( lambda ts : ts[0] ).reduce(sum_reduce_dict)
    stats_list = results_rdd.map( lambda ts: ts[1]).collect()  # Per-image stats
    stats = dict(zip(img_keys,stats_list))  # Restore list to keyed format

    final_stats = get_final_stats( totals, task, use_tightness )

    return final_stats, stats


def evaluate_image_tuple(gt_pred, task, can_match, score_match):
    """Pickle-able top-level function accepting paired lead inputs as a tuple"""
    return evaluate_image(gt_pred[0], gt_pred[1], task,
                          can_match, score_match )


def pool_evaluate(gt: dict[str,ImageData],
                  pred: dict[str,ImageData],
                  task: str,
                  can_match: Callable[[WordData,WordData,float],
                                      bool],
                  score_match: Callable[[WordData,WordData,float],
                                        bool],
                  use_tightness: bool) \
                   -> Tuple[dict[str,float], dict[str,dict[str,float]]]:
    """Run the primary evaluation protocol in parallel using Pool

    Returns:
      final_stats : dict containing pooled statistics for the entire data set
      stats : dict containing statistics for each image in the data set
    """

    img_keys,data = flatten_zip_dict(gt,pred,task)  # zip image keys for map

    # Fill in arguments for pickle-able function to use with Pool.map
    simple_evaluate = partial(evaluate_image_tuple, task=task,
                              can_match=can_match, score_match=score_match )

    with Pool() as pool:
        results = pool.map( simple_evaluate, data)

    totals = reduce( sum_reduce_dict, [ts[0] for ts in results] )
    # Restore list to keyed format
    stats = dict(zip(img_keys,[ts[1] for ts in results]))

    final_stats = get_final_stats( totals, task, use_tightness )

    return final_stats, stats


# NB: Prefer these functions to be local to configure_protocol, but they must
# be top level, in order to be pickleable for multiprocessing.Pool

# Using epsilon for ignore ground truths biases toward valid matches
def det_score(g: WordData, d: WordData,
                  iou: float, use_tightness: Optional[bool] =True) -> float:
    """Detection and detection+linking can optimize for Fscore, or PQ
    by using IoU"""
    return IGNORE_EPSILON if g['ignore'] else (iou if use_tightness else 1.0)


def rec_score(g: WordData, d: WordData,
              iou: float, use_tightness: Optional[bool] =True) -> float:
    """Recognition or Recognition+Linking can approximately optimize for Hmean with or withought tightness by CNED or CNED*IoU"""
    if g['ignore']:
        return IGNORE_EPSILON
    return str_score(g['text'], d['text']) * (iou if use_tightness else 1.0)


def det_valid(g: WordData, d: WordData,
              iou: float, match_thresh: float) -> bool:
    """Detections (linking or not) require IoU threshold to be met"""
    return iou > match_thresh


def rec_valid(g: WordData, d: WordData,
              iou: float, match_thresh: float) -> bool:
    """Recognitions require minimum IoU threshold and string matches.
    Allow matches to an ignore without requiring string matching; their
    score will be lower than matches to non-ignores.
    """
    return iou > match_thresh and (g['ignore'] or g['text'] == d['text'])


def str_score(gs: str, ds: str) -> float:
    """Complementary normalized edit distance, 1-NED"""
    return 1.0 - normalized_levenshtein(gs,ds)


def configure_protocol(task: str, match_thresh: float,
                       use_tightness: bool) -> \
    Tuple[Callable[[WordData,WordData,float],bool],
          Callable[[WordData,WordData,float],float]]:
    """Process arguments to configure specific protocol functionality: string
    match (bool) and score (i.e., 1-NED) functions as well as
    correspondence candidate criteria (bool) and match score (float)
    functions.

    Parameters
      task:          Specifies task to configure
      match_thresh:  IoU required to be considered a valid match (0-1).
      use_tightness: Whether to include tightness among the comprehensive 
                       evaluation score terms
    Returns
      can_match:     Predicate taking ground truth and predicted word dicts with
                       their pre-calculated iou score and returning whether the
                       correspondence satisfies match criteria
      score_match:   Function taking ground truth and predicted word dicts with
                       their pre-calculated iou score and returning their match
                       score (assumes they are valid matches)
    """

    # Since character accuracy is used in soft scoring, use det_valid for
    # matching, rather than rec_valid (which requires exact string match)
    can_match   = partial(det_valid, match_thresh=match_thresh)
    
    if 'rec' in task:
        score_match = partial(rec_score, use_tightness=use_tightness)
    else:
        score_match = partial(det_score, use_tightness=use_tightness)

    return can_match, score_match


def main():
    """Main entry point for evaluation script"""

    args = parser.parse_args()

    is_e2e = 'rec' in args.task

    gt_anno = load_ground_truth( args.gt, is_e2e=is_e2e,
                                 image_regex=args.gt_regex )
    preds = load_predictions( args.pred, is_e2e=is_e2e,
                              image_regex=args.gt_regex )

    # Verify we have the same images (key sets)
    if gt_anno.keys() != preds.keys() :
        warn_image_keys( gt_anno.keys(), preds.keys() )

    if args.parallel != 'none':
        # Introduction of edge statistics has broken parallel methods, because
        # the adjacency matrix calculations require non-flattened data.
        # This fixable limitation should be addressed.
        raise NotImplementedError('Parallel methods not yet supported')
    
    if args.parallel == 'spark':
        eval_fn = spark_evaluate
    elif args.parallel == 'pool':
        eval_fn = pool_evaluate
    else:
        eval_fn = evaluate

    can_match, score_match = configure_protocol(args.task, args.iou_threshold,
                                                args.use_tightness)
    
    overall,per_image = eval_fn( gt_anno, preds,
                                 args.task, can_match, score_match,
                                 args.use_tightness)

    print(overall)

    if args.output:
        with open(args.output,'w',encoding='utf-8') as fd:
            json.dump( {'images': per_image,
                        'results': overall }, fd, indent=4 )


if __name__ == "__main__":
    main()
