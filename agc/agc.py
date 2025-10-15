#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""OTU clustering"""

import argparse
import sys
import os
import gzip
import statistics
import textwrap
from pathlib import Path
from collections import Counter
from typing import Iterator, Dict, List
# https://github.com/briney/nwalign3
# ftp://ftp.ncbi.nih.gov/blast/matrices/
import nwalign3 as nw

# --- Compatibilité NumPy 2.0+ : réintroduire les alias supprimés ---
import numpy as np

# Compat NumPy 2.0+ : ne patcher QUE np.int, sans accéder aux autres alias dépréciés
if 'int' not in np.__dict__:
    np.int = int  # type: ignore[attr-defined]


__author__ = "Laura DUFOUR"
__copyright__ = "Universite Paris Cite"
__credits__ = ["Laura DUFOUR"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Laura DUFOUR"
__email__ = "laura.dufour@etu.u-paris.fr"
__status__ = "Developpement"



def isfile(path: str) -> Path:  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments(): # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, usage=
                                     "{0} -h"
                                     .format(sys.argv[0]))
    parser.add_argument('-i', '-amplicon_file', dest='amplicon_file', type=isfile, required=True, 
                        help="Amplicon is a compressed fasta file (.fasta.gz)")
    parser.add_argument('-s', '-minseqlen', dest='minseqlen', type=int, default = 400,
                        help="Minimum sequence length for dereplication (default 400)")
    parser.add_argument('-m', '-mincount', dest='mincount', type=int, default = 10,
                        help="Minimum count for dereplication  (default 10)")
    parser.add_argument('-o', '-output_file', dest='output_file', type=Path,
                        default=Path("OTU.fasta"), help="Output file")
    return parser.parse_args()


def read_fasta(amplicon_file: Path, minseqlen: int) -> Iterator[str]:
    """Read a compressed fasta and extract all fasta sequences.

    :param amplicon_file: (Path) Path to the amplicon file in FASTA.gz format.
    :param minseqlen: (int) Minimum amplicon sequence length
    :return: A generator object that provides the Fasta sequences (str).
    """
    with gzip.open(amplicon_file, "rt") as monfich:
        seq = ""
        for line in monfich:
            line = line.strip()
            if line is None:
                continue
            if not line.startswith(">"):
                seq += line.upper()
            else:
                if len(seq) >= minseqlen:
                    yield seq
                seq = ""
        if len(seq) >= minseqlen:
            yield seq
            


def dereplication_fulllength(amplicon_file: Path, minseqlen: int, mincount: int) -> Iterator[List]:
    """Dereplicate the set of sequence

    :param amplicon_file: (Path) Path to the amplicon file in FASTA.gz format.
    :param minseqlen: (int) Minimum amplicon sequence length
    :param mincount: (int) Minimum amplicon count
    :return: A generator object that provides a (list)[sequences, count] of sequence with a count >= mincount and a length >= minseqlen.
    """
    dic_uniqseq = {}

    for seq in read_fasta(amplicon_file,minseqlen):
        if seq not in dic_uniqseq:
            dic_uniqseq[seq] = 1
        else:
            dic_uniqseq[seq] += 1
    
    # on stocke les clés ordonnées en output de sorted dans une liste
    ordered_seqlist = sorted(dic_uniqseq, key = dic_uniqseq.get, reverse = True) 

    for seq in ordered_seqlist:
        if dic_uniqseq[seq] >= mincount:
            yield [seq, dic_uniqseq[seq]]


def get_identity(alignment_list: List[str]) -> float:
    """Compute the identity rate between two sequences

    :param alignment_list:  (list) A list of aligned sequences in the format ["SE-QUENCE1", "SE-QUENCE2"]
    :return: (float) The rate of identity between the two sequences.
    """
    seq1 = alignment_list[0]
    seq2 = alignment_list[1]

    nb_same_nct = 0 # nombre de nucléotides identiques
    for idx, nclt_s1 in enumerate(seq1):
        if nclt_s1 == seq2[idx]:
            nb_same_nct += 1
    score_id = 100 * (nb_same_nct/len(max(alignment_list)))

    return score_id

def abundance_greedy_clustering(amplicon_file: Path, minseqlen: int, mincount: int, chunk_size: int, kmer_size: int) -> List:
    """Compute an abundance greedy clustering regarding sequence count and identity.
    Identify OTU sequences.

    :param amplicon_file: (Path) Path to the amplicon file in FASTA.gz format.
    :param minseqlen: (int) Minimum amplicon sequence length.
    :param mincount: (int) Minimum amplicon count.
    :param chunk_size: (int) A fournir mais non utilise cette annee
    :param kmer_size: (int) A fournir mais non utilise cette annee
    :return: (list) A list of all the [OTU (str), count (int)] .
    """
    otu_list: List = []
    subs_matrix_path = str(Path(__file__).parent / "MATCH")

    gen = dereplication_fulllength(amplicon_file, minseqlen, mincount)

    # 1) Seed: on prend la séquence la plus abondante comme première OTU
    try:
        seq0, c0 = next(gen)
        otu_list.append([seq0, c0])
    except StopIteration:
        return otu_list  # aucun candidat

    # 2) Pour chaque autre séquence, comparer aux OTU existantes
    for sequence, count in gen:
        is_new = True
        for otu_seq, _ in otu_list:
            aln = nw.global_align(sequence, otu_seq, gap_open=-1, gap_extend=-1,
                                  matrix=subs_matrix_path)
            if get_identity(aln) >= 97.0:
                is_new = False
                break
        if is_new:
            otu_list.append([sequence, count])

    return otu_list




def write_OTU(OTU_list: List, output_file: Path) -> None:
    """Write the OTU sequence in fasta format.

    :param OTU_list: (list) A list of OTU sequences
    :param output_file: (Path) Path to the output file
    """
    with output_file.open("w") as out:
        for idx, (seq, count) in enumerate(OTU_list, start=1):
            out.write(f">OTU_{idx} occurrence:{count}\n")
            out.write(textwrap.fill(seq, width=80) + "\n")



def main():
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()
    # Votre programme ici

    otu_list = abundance_greedy_clustering(
        amplicon_file=args.amplicon_file,
        minseqlen=args.minseqlen,
        mincount=args.mincount,
        chunk_size=100,  # non utilisé cette année
        kmer_size=8      # non utilisé cette année
    )

    write_OTU(otu_list, args.output_file)


if __name__ == '__main__':
    main()
