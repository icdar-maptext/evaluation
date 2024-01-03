"""Command-line script for standalone ICDAR map-text competition evaluation

Copyright 2024 Jerod Weinman

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

from functools import reduce, partial
from multiprocessing import Pool
from typing import Union, Tuple, Any, Optional

import scipy  # type: ignore
import numpy as np
import numpy.typing as npt
import shapely  # type: ignore

from pyeditdistance.distance import normalized_levenshtein  # type: ignore


parser = argparse.ArgumentParser(
    description='Map Text Competition Task Evaluation')
parser.add_argument('--gt', type=str, required=True,
                    help="Path to the ground truth JSON file")
parser.add_argument('--pred', type=str, required=True,
                    help="Path to the predictions JSON file")
parser.add_argument('--output', type=str, required=False, default=None,
                    help="Path to the JSON file containing results")
parser.add_argument('--task', type=str, required=True,
                    choices=['det', 'detlink', 'detrec', 'detreclink'],
                    help="Task to evaluate against")
parser.add_argument('--iou-threshold', type=float, default=0.5,
                    help="Minimum IoU for elements to be considered a match")
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


def add_group_info( data: dict[str,ImageData],
                    is_e2e: bool,
                    is_gt: bool = True):
    """Add to each group the following fields
     'geometry': union of all words' polygons in the group

    When is_e2e
     'text': space-separated concatenation of words' text

    When is_gt
     'ignore': Disjunction of words' ignore field

    Returns
      Nothing; called for side-effect (mutating data)
    """
    for image_groups in data.values():
        for group in image_groups:
            words = group['words']
            polys = [ word['geometry'] for word in words ]
            group['geometry'] = shapely.unary_union(polys)
            if is_e2e:
                group['text'] = ' '.join( [ word['text'] for word in words
                                            if 'text' in word] )
            if is_gt:
                group['ignore'] = any( word['ignore'] for word in words )


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
                       is_linking: bool,
                       is_e2e: bool,
                       image_regex: Optional[str] = None,
                       verify: bool = True ) -> dict[str,ImageData]:
    """ Load the ground truth file and optionally verify contents format

    Arguments
      gt_file : Path to the ground truth JSON file (see competition format)
      is_linking : Whether the evaluation includes linking
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

    # Re-index: image --> groups, wrapping group lists to dicts (for more fields)
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
    if is_linking:
        add_group_info(gt, is_e2e=is_e2e, is_gt=True)

    return gt


def load_predictions( preds_file: str,
                      is_linking: bool,
                      is_e2e: bool,
                      image_regex: Optional[str] = None ) -> dict[str,ImageData]:
    """ Load the predictions file and verify contents format

    Arguments
      preds_file : Path to the predictions JSON file (see competition format)
      is_linking : Whether the evaluation includes linking
      is_e2e: Whether the evaluation is end-to-end (i.e., includes recognition)
    Returns
      preds : Dict indexed by the image id, giving the list of groups
    """

    logging.info('Loading predictions...')

    with open(preds_file, encoding='utf-8') as fd:
        preds_raw = json.load(fd)

    verify_predictions_format(preds_raw, is_e2e)

    # Re-index: image --> groups, wrapping group lists to dicts (for more fields)
    preds = {}
    regex = re.compile(image_regex) if image_regex else None
    for entry in preds_raw:
        if regex and not regex.match(entry['image']):  # Skip if no match
            continue
        groups = [ { 'words' : group } for group in entry['groups'] ]
        preds[entry['image']] = groups

    # Add additional/transformed fields for evaluation processing
    add_word_polygons(preds)
    if is_linking:
        add_group_info(preds, is_e2e=is_e2e, is_gt=False)

    return preds


def calc_iou_pairs( gt: Union[list[WordData],list[GroupData]],
                    pred: Union[list[WordData],list[GroupData]],
                    match_thresh: float ) -> npt.ArrayLike:
    """Return the IoU between all pairs of shapes.

    Arguments
      gt :  List of dicts containing ground truth elements (each has the field
           'geometry' among others).
      pred : List of dicts containing predicted elements (each has the field
             'geometry' among others).
      match_thresh : Threshold the calculated IoU must exceed or else be
                     trimmed down to 0.
    Returns
      ious : MxN numpy float array, where M is len(gt) and N is len(pred).
    """

    def calc_iou( p, q ) -> float :
        """ Return the IoU between two shapes """
        if p.intersects(q):
            intersection = p.intersection(q).area
            union = p.union(q).area
            return intersection / union
        else:
            return 0.0


    ious = np.zeros( (len(gt),len(pred)), dtype=np.double )

    for i,gt_el in enumerate(gt):
        for j,pred_el in enumerate(pred):
            try:
                the_iou = calc_iou( gt_el['geometry'], pred_el['geometry'])
            except shapely.errors.GEOSException as e:
                logging.warning('Error at iou(%d,%d): %s}. Skipping ...',i,j,e)
                continue
            if the_iou != 0 and the_iou > match_thresh:
                ious[i,j] = the_iou
    return ious


def calc_transcriptions_correct ( gt: Union[list[WordData],list[GroupData]],
                                  pred: Union[list[WordData],list[GroupData]] ) \
                                  -> npt.ArrayLike: # list[bool]:
    """Calculate where paired ground truth and prediction texts meet
     the criteria for correctness.

    Arguments
      gt : Ground truth elements, with field 'text'
      pred : Corresponding predicted elements, with field 'text'
    Preconditions
      len(gt)==len(pred), so that elements are in correspondence
    Returns
       boolean NumPy array indicating corresponding pairs in list with
         "matching" text fields
    """

    # simple local function for now, imagining a future where it is a parameter
    def is_match(g: dict[str,Any] ,p: dict[str,Any]) -> bool:
        return g['text'] == p['text']

    # TODO(jjw) generalize matching criteria?
    results = [ is_match(g,p) for (g,p) in zip(gt,pred)]
    return np.asarray(results, dtype=np.bool_)  # np'd for indexing


def calc_transcriptions_scores ( gt: Union[list[WordData],list[GroupData]],
                                 pred: Union[list[WordData],list[GroupData]] ) \
                                 -> npt.ArrayLike:  # list[float]:
    """Calculate transcription scores between paired ground truth and prediction
    texts (i.e., 1 - normalized edit distance).

    Arguments
      gt : Ground truth elements, with field 'text'
      pred : Corresponding predicted elements, with field 'text'
    Preconditions
      len(gt)==len(pred), so that elements are in correspondence
    Returns
       float NumPy array scoring the corresponding pairs' 'text' fields
         "matching" text fields
    """

    # local function for now, imagine a future where it is a parameter
    def score_pred(g: str ,p: str) -> float:
        return 1 - normalized_levenshtein(g,p)

    # TODO(jjw) generalize score function?
    results = [ score_pred(g['text'],p['text']) for (g,p) in zip(gt,pred)]
    return np.asarray(results)


def get_stats( num_tp: Number, num_gt: Number, num_pred: Number, tot_iou: Number,
               prefix: str = '') -> dict[str,float] :
    """Calculate statistics: recall, precision, fscore, tightness, and quality
    from accumulated totals.

    Arguments
      num_tp : Number of true positives
      num_gt : Number of ground truth positives in the evaluation
      num_pred : Number of predicted positives in the evaluation
      tot_iou : Total IoU scores among true positives
      prefix : Optional prefix for return result keys (default='')
    Returns
      dict containing statistics with keys 'recall', 'precision', 'fscore',
        'tightness' (average IoU score), and 'quality' (product of fscore and
        tightness).
    """
    recall    = float(num_tp) / num_gt   if (num_gt > 0)   else 0.0
    precision = float(num_tp) / num_pred if (num_pred > 0) else 0.0
    tightness = tot_iou / float(num_tp)  if (num_tp > 0)   else 0.0
    fscore    = 2.0*recall*precision / (recall+precision) \
        if (recall + precision > 0) else 0.0
    quality = tightness * fscore

    stats = {prefix+'recall'    : recall,
             prefix+'precision' : precision,
             prefix+'fscore'    : fscore,
             prefix+'tightness' : tightness,
             prefix+'quality'   : quality }
    return stats


def get_final_stats(totals: dict[str,Number],
                    task : str) -> dict[str,Number] :
    """Process totals to produce final statistics for the entire data set.

    Arguments
      totals : Dict with keys 'tp_det', 'total_gt', 'total_pred',
                 'total_det_tightness', and (if 'rec' in task), 'tp_rec',
                 'total_rec_tightness', and 'total_rec_score'.
      task : String containing a valid task (cf parser)
    Returns
      dict containing statistics with keys 'det_recall', 'det_precision',
        'det_fscore', 'det_tightness' (average IoU score),  'det_quality'
        (product of fscore and tightness), and (if 'rec' in task) 'rec_recall',
        'rec_precision', 'rec_fscore', 'rec_tightness',  'rec_quality',
       'char_accuracy' and 'char_quality' (product of det_quality and
       char_accuracy).
    """
    final_stats = get_stats( totals['tp_det'],
                             totals['total_gt'],
                             totals['total_pred'],
                             totals['total_det_tightness'], 'det_')
    if 'rec' in task:
        final_stats.update( get_stats( totals['tp_rec'],
                                       totals['total_gt'],
                                       totals['total_pred'],
                                       totals['total_rec_tightness'], 'rec_') )
        if totals['tp_det'] > 0:
            accuracy = totals['total_rec_score'] / float(totals['tp_det'])
        else:
            accuracy = 0.0
        final_stats['char_accuracy'] = accuracy
        final_stats['char_quality'] = accuracy * final_stats['det_quality']

    return final_stats


def evaluate_image( gt: Union[list[WordData],list[GroupData]],
                    pred: Union[list[WordData],list[GroupData]],
                    task: str,
                    match_thresh: float = 0.5) -> Tuple[dict[str,float], dict[str,float]]:
    """Apply the appropriate evaluation scheme to lists of ground truth and
    prediction elements from the same image.

    Arguments
      gt : List of dicts containing ground truth elements (each has the fields
           'geometry', 'text', and 'ignore').
      pred : List of dicts containing predicted elements for evaluation (each
             has the fields 'geometry' and (if task contains 'rec') 'text' .
      task : string describing the task (det, detlink, detrec, detreclink)
      match_thresh : Minimum required IoU between ground truth and predicted
                     areas to be considered a match
    Returns
      results : dict containing totals for the accumulator
      stats : dict containing statistics for this image
    """

    def find_matches(ious):
        """Optimize the bipartite matches and filter them to allowable IoUs"""
        matches_gt,matches_pred = \
            scipy.optimize.linear_sum_assignment(ious, maximize=True)
        # Filter out misleading "matches" assigned with an IoU of 0
        matches_ious  = ious[matches_gt,matches_pred]
        matches_valid = matches_ious > match_thresh  # enforce threshold
        matches_ious  = matches_ious[matches_valid]
        matches_gt    = matches_gt  [matches_valid]
        matches_pred  = matches_pred[matches_valid]

        return matches_gt, matches_pred, matches_ious


    def eval_recognition( matches_gt, matches_pred, matches_ious, matches_count ):
        """Calculate the number of true positives and tightness/accuracy scores
        according to word recognition criteria"""
        # sub-lists of just the non-ignorable matching elements
        gt_matches   = [ gt[i]   for i in matches_gt  [matches_count] ]
        pred_matches = [ pred[j] for j in matches_pred[matches_count] ]
        # find and count the number of correct text predictions (i.e., 1-WER)
        correct_matches = calc_transcriptions_correct( gt_matches, pred_matches )
        tp_rec = np.sum(correct_matches)
        # tally tightness only among correctly recognized, non-ignorable texts
        total_rec_tightness = np.sum(matches_ious[matches_count][correct_matches])
        # measure the text (mis)prediction (i.e., 1-CER)
        text_score_matches = calc_transcriptions_scores(gt_matches, pred_matches)
        # tally scores among all non-ignorable texts
        total_rec_score = np.sum( text_score_matches )

        return tp_rec, total_rec_tightness, total_rec_score

    ious = calc_iou_pairs( gt, pred, match_thresh )
    matches_gt, matches_pred, matches_ious = find_matches(ious)

    # Mark as ignorable any predicted regions that matched an ignored region
    matches_ignore = np.asarray([gt[i]['ignore'] for i in matches_gt])
    matches_count  = np.logical_not(matches_ignore)
    num_matches_ignore = np.sum(matches_ignore)

    # Count the total number of ground truth entries marked as ignore
    num_gt_ignore  = len( [ el for el in gt if el['ignore'] ] )

    total_pred = len(pred)  - num_matches_ignore
    total_gt   = len(gt)    - num_gt_ignore

    tp_det = len(matches_pred) - num_matches_ignore
    # assert tp_det == np.sum(matches_count)
    total_det_tightness = np.sum(matches_ious[matches_count])

    results = { 'tp_det' : tp_det,
                'total_gt' : total_gt,
                'total_pred' :  total_pred,
                'total_det_tightness' : total_det_tightness }
    # Calculate detection statistics (also used for character-level recognition)
    stats = get_stats( tp_det, total_gt, total_pred, total_det_tightness, 'det_')

    if 'rec' in task:
        tp_rec, total_rec_tightness, total_rec_score = eval_recognition(
            matches_gt, matches_pred, matches_ious, matches_count )

        # Calculate recognition statistics (i.e., correct transcriptions)
        stats_rec = get_stats( tp_rec, total_gt, total_pred, total_rec_tightness,
                               'rec_' )
        stats.update( stats_rec )  # Merge recognition stats

        accuracy = total_rec_score / float(tp_det) if (tp_det>0) else 0.0
        stats['char_accuracy'] = accuracy
        stats['char_quality'] = accuracy * stats['det_quality']

        results_rec = { 'tp_rec' : tp_rec,
                        'total_rec_tightness' : total_rec_tightness,
                        'total_rec_score' : total_rec_score }
        results.update(results_rec)

    return results, stats


def evaluate(gt: dict[str,ImageData],
             pred: dict[str,ImageData],
             task: str,
             match_thresh: float) -> Tuple[dict[str,float], dict[str,dict[str,float]]]:
    """Run the primary evaluation rubric

    Returns:
      final_stats : dict containing pooled statistics for the entire data set
      stats : dict containing statistics for each image in the data set
    """

    def accumulate( totals: dict[str,float], results: dict[str,float] ):
        """Side-effect totals by accumulating matching keys of results"""
        for (k,v) in results.items():
            totals[k] += v

    # initialize accumulator
    totals = { 'tp_det' : 0.,
               'total_pred' : 0.,
               'total_gt' : 0.,
               'total_det_tightness' : 0.,
               'tp_rec' : 0.,
               'total_rec_tightness' : 0.,
               'total_rec_score' : 0. }
    stats = {} # Collected per-image statistics

    for (img,gt_groups) in gt.items():  # Process each image
        pred_groups = pred[img] if img in pred else []

        if 'link' in task:  # evaluate at top level (groups)
            img_results, img_stats = evaluate_image( gt_groups, pred_groups,
                                                     task, match_thresh )
        else:
            # flatten to list of words for evaluation
            gt_words = [ word for group in gt_groups
                         for word in group['words'] ]
            pred_words = [ word for group in pred_groups
                           for word in group['words'] ]

            img_results, img_stats = evaluate_image( gt_words, pred_words,
                                                     task, match_thresh )

        accumulate(totals,img_results)
        stats[img] = img_stats

    final_stats = get_final_stats( totals, task )  # Process totals
    return final_stats, stats


def sum_reduce_dict( a: dict[str,float], b: dict[str,float] ) -> dict[str,float]:
    """Reduce dictionaries by summing matching keys"""
    return { k : v + b[k] for (k,v) in a.items() }


def flatten_zip_dict( gt: dict[str,ImageData],
                      pred: dict[str,ImageData],
                      task: str) -> Tuple[list[str],list[Tuple[ImageData,ImageData]]]:
    """Helper for parallel processing functions. Zips corresponding ground truth
    and prediction entries into list of tuples, according to the task."""
    img_keys = list(gt.keys())  # cache keys to be certain of fixed ordering

    # Reformulate gt,pred dicts as list of tuples (for parallelized data frame)
    data = [ (gt[img], pred[img] if img in pred else [])
             for img in img_keys ]

    if 'link' not in task:  # flatten entries to lists of words for evaluation
        data = [ ( [word for group in g for word in group['words'] ] ,
                   [word for group in p for word in group['words'] ])
                 for (g,p) in data ]
    return img_keys,data


def spark_evaluate(gt: dict[str,ImageData],
                   pred: dict[str,ImageData],
                   task: str,
                   match_thresh: float,
                   images_per_slice: int = 10 ) -> \
                   Tuple[dict[str,float], dict[str,dict[str,float]]]:
    """Run the primary evaluation rubric in parallel using Apache Spark

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
    data_rdd = spark_session.sparkContext.parallelize(data, numSlices=num_slices)
    # Parallel run: produces list of tuples: [(totals, image_stats), ...]
    results_rdd = data_rdd.map( lambda gp  : evaluate_image(gp[0], gp[1],
                                                            task, match_thresh) )
    results_rdd.persist() # Cache results to avoid subsequent re-computation
    # Splice totals and reduce by summing, then splice per-image stats
    totals = results_rdd.map( lambda ts : ts[0] ).reduce(sum_reduce_dict)
    stats_list = results_rdd.map( lambda ts: ts[1]).collect()  # Per-image stats
    stats = dict(zip(img_keys,stats_list))  # Restore list to keyed format

    final_stats = get_final_stats( totals, task )

    return final_stats, stats


def evaluate_image_tuple(gt_pred, task, match_thresh):
    """Pickle-able top-level function accepting paired lead inputs as a tuple"""
    return evaluate_image(gt_pred[0], gt_pred[1], task, match_thresh)


def pool_evaluate(gt: dict[str,ImageData],
                  pred: dict[str,ImageData],
                  task: str,
                  match_thresh: float ) -> \
                  Tuple[dict[str,float], dict[str,dict[str,float]]]:
    """Run the primary evaluation rubric in parallel using multiprocessing Pool

    Returns:
      final_stats : dict containing pooled statistics for the entire data set
      stats : dict containing statistics for each image in the data set
    """

    img_keys,data = flatten_zip_dict(gt,pred,task)  # zip image keys for map

    # Fill in arguments for pickle-able function to use with Pool.map
    simple_evaluate = partial(evaluate_image_tuple, task=task, match_thresh=match_thresh)

    with Pool() as pool:
        results = pool.map( simple_evaluate, data)

    totals = reduce( sum_reduce_dict, [ts[0] for ts in results] )
    stats = dict(zip(img_keys,[ts[1] for ts in results]))  # Restore list to keyed format

    final_stats = get_final_stats( totals, task )

    return final_stats, stats


def main():
    """Main entry point for evaluation script"""

    args = parser.parse_args()
    # TODO(jjw) Add command-line options to configure logger?

    is_linking = 'link' in args.task
    is_e2e     = 'rec' in args.task

    gt_anno = load_ground_truth( args.gt, is_linking=is_linking, is_e2e=is_e2e,
                                 image_regex=args.gt_regex )
    preds = load_predictions( args.pred, is_linking=is_linking, is_e2e=is_e2e,
                              image_regex=args.gt_regex )

    # Verify we have the same images (key sets)
    if gt_anno.keys() != preds.keys() :
        warn_image_keys( gt_anno.keys(), preds.keys() )

    if args.parallel == 'spark':
        eval_fn = spark_evaluate
    elif args.parallel == 'pool':
        eval_fn = pool_evaluate
    else:
        eval_fn = evaluate

    overall,per_image = eval_fn( gt_anno, preds,
                                 args.task, args.iou_threshold )

    print(overall)

    if args.output:
        with open(args.output,'w',encoding='utf-8') as fd:
            json.dump( {'images': per_image,
                        'results': overall }, fd, indent=4 )


if __name__ == "__main__":
    main()
