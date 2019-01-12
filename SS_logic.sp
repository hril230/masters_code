%%%%%%%%%%%%%%%%%%%%%%%%%
sorts
%%%%%%%%%%%%%%%%%%%%%%%%%
#blocks = 2..5.
#feature = {narrow_base, lean, block_displaced, stable, unstable}.
#question = 0..3.
#answer = 0..49.
#rule = 0..7.

%%%%%%%%%%%%%%%%%%%%%%%%
predicates
%%%%%%%%%%%%%%%%%%%%%%%%
num_blocks(#blocks).
structure(#feature).
question(#question).
answer(#answer).
rule(#rule).

%%%%%%%%%%%%%%%%%%%%%%%%
rules
%%%%%%%%%%%%%%%%%%%%%%%%

%% Closed world assumptions
-num_blocks(I) :- num_blocks(J), I!=J.
-structure(narrow_base) :- not structure(narrow_base).
-structure(lean) :- not structure(lean).
-structure(block_displaced) :- not structure(block_displaced).
-answer(A) :- not answer(A).

%% Only one answer can be true
:- answer(A1), answer(A2), A1!=A2.


%% Instability rules
rule(0) :- num_blocks(4).
rule(1) :- num_blocks(5).
rule(2) :- structure(narrow_base).
rule(3) :- structure(lean).
rule(4) :- structure(block_displaced).
structure(unstable) :- rule(0).
structure(unstable) :- rule(1).
structure(unstable) :- rule(2).
structure(unstable) :- rule(3).
structure(unstable) :- rule(4).


%% Stability rules
rule(5) :- num_blocks(2), -structure(lean).
rule(6) :- num_blocks(2), -structure(block_displaced).
rule(7) :- -structure(lean), -structure(block_displaced).
structure(stable) :- rule(5).
structure(stable) :- rule(6).
structure(stable) :- rule(7).


%% Logic for question answering
% answers to question zero (is this structure stable?)
answer(0) :- structure(unstable), question(0).
answer(1) :- structure(stable), question(0).
% answers to question one (what is making this structure unstable?)
answer(2) :- structure(stable), question(1).
answer(3) :- structure(unstable), num_blocks(4), -structure(narrow_base), -structure(lean), -structure(block_displaced), question(1).
answer(3) :- structure(unstable), num_blocks(5), -structure(narrow_base), -structure(lean), -structure(block_displaced), question(1).
answer(4) :- structure(unstable), num_blocks(2), structure(narrow_base), -structure(lean), -structure(block_displaced), question(1).
answer(4) :- structure(unstable), num_blocks(3), structure(narrow_base), -structure(lean), -structure(block_displaced), question(1).
answer(5) :- structure(unstable), num_blocks(2), -structure(narrow_base), structure(lean), -structure(block_displaced), question(1).
answer(5) :- structure(unstable), num_blocks(3), -structure(narrow_base), structure(lean), -structure(block_displaced), question(1).
answer(6) :- structure(unstable), num_blocks(2), -structure(narrow_base), -structure(lean), structure(block_displaced), question(1).
answer(6) :- structure(unstable), num_blocks(3), -structure(narrow_base), -structure(lean), structure(block_displaced), question(1).
answer(7) :- structure(unstable), num_blocks(4), structure(narrow_base), -structure(lean), -structure(block_displaced), question(1).
answer(7) :- structure(unstable), num_blocks(5), structure(narrow_base), -structure(lean), -structure(block_displaced), question(1).
answer(8) :- structure(unstable), num_blocks(4), -structure(narrow_base), structure(lean), -structure(block_displaced), question(1).
answer(8) :- structure(unstable), num_blocks(5), -structure(narrow_base), structure(lean), -structure(block_displaced), question(1).
answer(9) :- structure(unstable), num_blocks(4), -structure(narrow_base), -structure(lean), structure(block_displaced), question(1).
answer(9) :- structure(unstable), num_blocks(5), -structure(narrow_base), -structure(lean), structure(block_displaced), question(1).
answer(10) :- structure(unstable), num_blocks(2), structure(narrow_base), structure(lean), -structure(block_displaced), question(1).
answer(10) :- structure(unstable), num_blocks(3), structure(narrow_base), structure(lean), -structure(block_displaced), question(1).
answer(11) :- structure(unstable), num_blocks(2), structure(narrow_base), -structure(lean), structure(block_displaced), question(1).
answer(11) :- structure(unstable), num_blocks(3), structure(narrow_base), -structure(lean), structure(block_displaced), question(1).
answer(12) :- structure(unstable), num_blocks(2), -structure(narrow_base), structure(lean), structure(block_displaced), question(1).
answer(12) :- structure(unstable), num_blocks(3), -structure(narrow_base), structure(lean), structure(block_displaced), question(1).
answer(13) :- structure(unstable), num_blocks(4), structure(narrow_base), structure(lean), -structure(block_displaced), question(1).
answer(13) :- structure(unstable), num_blocks(5), structure(narrow_base), structure(lean), -structure(block_displaced), question(1).
answer(14) :- structure(unstable), num_blocks(4), structure(narrow_base), -structure(lean), structure(block_displaced), question(1).
answer(14) :- structure(unstable), num_blocks(5), structure(narrow_base), -structure(lean), structure(block_displaced), question(1).
answer(15) :- structure(unstable), num_blocks(4), -structure(narrow_base), structure(lean), structure(block_displaced), question(1).
answer(15) :- structure(unstable), num_blocks(5), -structure(narrow_base), structure(lean), structure(block_displaced), question(1).
answer(16) :- structure(unstable), num_blocks(2), structure(narrow_base), structure(lean), structure(block_displaced), question(1).
answer(16) :- structure(unstable), num_blocks(3), structure(narrow_base), structure(lean), structure(block_displaced), question(1).
answer(17) :- structure(unstable), num_blocks(4), structure(narrow_base), structure(lean), structure(block_displaced), question(1).
answer(17) :- structure(unstable), num_blocks(5), structure(narrow_base), structure(lean), structure(block_displaced), question(1).
answer(18) :- structure(unstable), num_blocks(2), -structure(narrow_base), -structure(lean), -structure(block_displaced), question(1).
answer(18) :- structure(unstable), num_blocks(3), -structure(narrow_base), -structure(lean), -structure(block_displaced), question(1).
% answers to question two (what is making this structure stable?)
answer(18) :- structure(stable), num_blocks(4), structure(narrow_base), structure(lean), structure(block_displaced), question(2).
answer(18) :- structure(stable), num_blocks(5), structure(narrow_base), structure(lean), structure(block_displaced), question(2).
answer(19) :- structure(unstable), question(2).
answer(20) :- structure(stable), num_blocks(2), structure(narrow_base), structure(lean), structure(block_displaced), question(2).
answer(20) :- structure(stable), num_blocks(3), structure(narrow_base), structure(lean), structure(block_displaced), question(2).
answer(21) :- structure(stable), num_blocks(4), -structure(narrow_base), structure(lean), structure(block_displaced), question(2).
answer(21) :- structure(stable), num_blocks(5), -structure(narrow_base), structure(lean), structure(block_displaced), question(2).
answer(22) :- structure(stable), num_blocks(4), structure(narrow_base), -structure(lean), structure(block_displaced), question(2).
answer(22) :- structure(stable), num_blocks(5), structure(narrow_base), -structure(lean), structure(block_displaced), question(2).
answer(23) :- structure(stable), num_blocks(4), structure(narrow_base), structure(lean), -structure(block_displaced), question(2).
answer(23) :- structure(stable), num_blocks(5), structure(narrow_base), structure(lean), -structure(block_displaced), question(2).
answer(24) :- structure(stable), num_blocks(2), -structure(narrow_base), structure(lean), structure(block_displaced), question(2).
answer(24) :- structure(stable), num_blocks(3), -structure(narrow_base), structure(lean), structure(block_displaced), question(2).
answer(25) :- structure(stable), num_blocks(2), structure(narrow_base), -structure(lean), structure(block_displaced), question(2).
answer(25) :- structure(stable), num_blocks(3), structure(narrow_base), -structure(lean), structure(block_displaced), question(2).
answer(26) :- structure(stable), num_blocks(2), structure(narrow_base), structure(lean), -structure(block_displaced), question(2).
answer(26) :- structure(stable), num_blocks(3), structure(narrow_base), structure(lean), -structure(block_displaced), question(2).
answer(27) :- structure(stable), num_blocks(4), -structure(narrow_base), -structure(lean), structure(block_displaced), question(2).
answer(27) :- structure(stable), num_blocks(5), -structure(narrow_base), -structure(lean), structure(block_displaced), question(2).
answer(28) :- structure(stable), num_blocks(4), -structure(narrow_base), structure(lean), -structure(block_displaced), question(2).
answer(28) :- structure(stable), num_blocks(5), -structure(narrow_base), structure(lean), -structure(block_displaced), question(2).
answer(29) :- structure(stable), num_blocks(4), structure(narrow_base), -structure(lean), -structure(block_displaced), question(2).
answer(29) :- structure(stable), num_blocks(5), structure(narrow_base), -structure(lean), -structure(block_displaced), question(2).
answer(30) :- structure(stable), num_blocks(2), -structure(narrow_base), -structure(lean), structure(block_displaced), question(2).
answer(30) :- structure(stable), num_blocks(3), -structure(narrow_base), -structure(lean), structure(block_displaced), question(2).
answer(31) :- structure(stable), num_blocks(2), -structure(narrow_base), structure(lean), -structure(block_displaced), question(2).
answer(31) :- structure(stable), num_blocks(3), -structure(narrow_base), structure(lean), -structure(block_displaced), question(2).
answer(32) :- structure(stable), num_blocks(2), structure(narrow_base), -structure(lean), -structure(block_displaced), question(2).
answer(32) :- structure(stable), num_blocks(3), structure(narrow_base), -structure(lean), -structure(block_displaced), question(2).
answer(33) :- structure(stable), num_blocks(4), -structure(narrow_base), -structure(lean), -structure(block_displaced), question(2).
answer(33) :- structure(stable), num_blocks(5), -structure(narrow_base), -structure(lean), -structure(block_displaced), question(2).
answer(34) :- structure(stable), num_blocks(2), -structure(narrow_base), -structure(lean), -structure(block_displaced), question(2).
answer(34) :- structure(stable), num_blocks(3), -structure(narrow_base), -structure(lean), -structure(block_displaced), question(2).
% answers to question three (what would need to be changed to make this structure stable?)
answer(2) :- structure(stable), question(3).
answer(18) :- structure(unstable), num_blocks(2), -structure(narrow_base), -structure(lean), -structure(block_displaced), question(3).
answer(18) :- structure(unstable), num_blocks(3), -structure(narrow_base), -structure(lean), -structure(block_displaced), question(3).
answer(35) :- structure(unstable), num_blocks(4), -structure(narrow_base), -structure(lean), -structure(block_displaced), question(3).
answer(35) :- structure(unstable), num_blocks(5), -structure(narrow_base), -structure(lean), -structure(block_displaced), question(3).
answer(36) :- structure(unstable), num_blocks(2), structure(narrow_base), -structure(lean), -structure(block_displaced), question(3).
answer(36) :- structure(unstable), num_blocks(3), structure(narrow_base), -structure(lean), -structure(block_displaced), question(3).
answer(37) :- structure(unstable), num_blocks(2), -structure(narrow_base), structure(lean), -structure(block_displaced), question(3).
answer(37) :- structure(unstable), num_blocks(3), -structure(narrow_base), structure(lean), -structure(block_displaced), question(3).
answer(38) :- structure(unstable), num_blocks(2), -structure(narrow_base), -structure(lean), structure(block_displaced), question(3).
answer(38) :- structure(unstable), num_blocks(3), -structure(narrow_base), -structure(lean), structure(block_displaced), question(3).
answer(39) :- structure(unstable), num_blocks(4), structure(narrow_base), -structure(lean), -structure(block_displaced), question(3).
answer(39) :- structure(unstable), num_blocks(5), structure(narrow_base), -structure(lean), -structure(block_displaced), question(3).
answer(40) :- structure(unstable), num_blocks(4), -structure(narrow_base), structure(lean), -structure(block_displaced), question(3).
answer(40) :- structure(unstable), num_blocks(5), -structure(narrow_base), structure(lean), -structure(block_displaced), question(3).
answer(41) :- structure(unstable), num_blocks(4), -structure(narrow_base), -structure(lean), structure(block_displaced), question(3).
answer(41) :- structure(unstable), num_blocks(5), -structure(narrow_base), -structure(lean), structure(block_displaced), question(3).
answer(42) :- structure(unstable), num_blocks(2), structure(narrow_base), structure(lean), -structure(block_displaced), question(3).
answer(42) :- structure(unstable), num_blocks(3), structure(narrow_base), structure(lean), -structure(block_displaced), question(3).
answer(43) :- structure(unstable), num_blocks(2), structure(narrow_base), -structure(lean), structure(block_displaced), question(3).
answer(43) :- structure(unstable), num_blocks(3), structure(narrow_base), -structure(lean), structure(block_displaced), question(3).
answer(44) :- structure(unstable), num_blocks(2), -structure(narrow_base), structure(lean), structure(block_displaced), question(3).
answer(44) :- structure(unstable), num_blocks(3), -structure(narrow_base), structure(lean), structure(block_displaced), question(3).
answer(45) :- structure(unstable), num_blocks(4), structure(narrow_base), structure(lean), -structure(block_displaced), question(3).
answer(45) :- structure(unstable), num_blocks(5), structure(narrow_base), structure(lean), -structure(block_displaced), question(3).
answer(46) :- structure(unstable), num_blocks(4), structure(narrow_base), -structure(lean), structure(block_displaced), question(3).
answer(46) :- structure(unstable), num_blocks(5), structure(narrow_base), -structure(lean), structure(block_displaced), question(3).
answer(47) :- structure(unstable), num_blocks(4), -structure(narrow_base), structure(lean), structure(block_displaced), question(3).
answer(47) :- structure(unstable), num_blocks(5), -structure(narrow_base), structure(lean), structure(block_displaced), question(3).
answer(48) :- structure(unstable), num_blocks(2), structure(narrow_base), structure(lean), structure(block_displaced), question(3).
answer(48) :- structure(unstable), num_blocks(3), structure(narrow_base), structure(lean), structure(block_displaced), question(3).
answer(49) :- structure(unstable), num_blocks(4), structure(narrow_base), structure(lean), structure(block_displaced), question(3).
answer(49) :- structure(unstable), num_blocks(5), structure(narrow_base), structure(lean), structure(block_displaced), question(3).

