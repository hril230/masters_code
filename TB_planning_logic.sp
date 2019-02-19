#const numSteps = 12.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  sorts
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#tower_level = 0..5.
#box_place = {blue_place, orange_place, yellow_place, green_place, red_place}.
#location = #tower_level + #box_place.
#box = {blue_box, orange_box, yellow_box, green_box, red_box}.
#step = 0..numSteps.

#inertial_fluent = loc(#box, #location) + in_hand(#box) + unstable_placement(#box) + empty(#tower_level).
#fluent = #inertial_fluent.

#action = pickup(#box) + place(#box, #tower_level).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
predicates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
goal_height(#tower_level).

holds(#fluent, #step).
occurs(#action, #step).

obs(#fluent, #step).
hpd(#action, #step).

success().
goal(#step).
something_happened(#step).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 rules
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Causal Laws %%

%% Grasping an object causes object to be in hand.
holds(in_hand(B), I+1) :- occurs(pickup(B), I).

%% Placing an object causes it to no longer be in hand.
-holds(in_hand(B), I+1) :- occurs(place(B, L), I).

%% Placing an object causes that object to occupy the level it was placed on (if the action completes successfully)
holds(loc(B, L), I+1) :- occurs(place(B, L), I), -holds(unstable_placement(B), I+1).

%% Placing an object causes that level to no longer be empty (if the action completes successfully)
-holds(empty(L), I+1) :- occurs(place(B, L), I), not holds(unstable_placement(B), I+1).



%% State Constraints %%

%% A box exists in only one location.
-holds(loc(B, L2), I) :- holds(loc(B, L1), I), L1!=L2.

%% Each location can only hold one box
-holds(loc(B1, L), I) :- holds(loc(B2, L), I), B1!=B2.

%% Only one object can be held at any time.
-holds(in_hand(B2), I) :- holds(in_hand(B1), I), B1 != B2.



%% Executability Conditions %%

%% Cannot place an object unless it is in hand.
-occurs(place(B, L), I) :-  -holds(in_hand(B), I).

%% Cannot pick up an object if it has something in hand.
-occurs(pickup(B1), I) :- holds(in_hand(B2), I).

%% Cannot pick up an object if you are not in the same room.
-occurs(pickup(R, O), I) :- holds(loc(R, L1), I),
			    holds(loc(O, L2), I), L1 != L2.

%% Cannot place a box on level zero
-occurs(place(B, 0), I).

%% Cannot place a box on top of an empty level (except level zero)
-occurs(place(B, L), I) :- holds(empty(L-1), I).

%% Cannot pick up a box if it is already part of the tower
-occurs(pickup(B), I) :- holds(loc(B, L), I), #tower_level(L).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Inertial axiom + CWA

%% General inertia axioms.
holds(F,I+1) :- #inertial_fluent(F),
                holds(F,I),
                not -holds(F,I+1).

-holds(F,I+1) :- #inertial_fluent(F),
                 -holds(F,I),
                 not holds(F,I+1).

%% CWA for Actions.
-occurs(A,I) :- not occurs(A,I).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Take what actually happened into account.
occurs(A,I) :- hpd(A,I).


%% Reality check axioms.
:- obs(F, I), -holds(F, I).

%% Awareness axiom.
holds(F, 0) | -holds(F, 0) :- #inertial_fluent(F).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Planning Module

%% Failure is not an option.
success :- goal(I).
:- not success.

%% Plan Actions minimally
occurs(A, I) :+ not goal(I).

%% Cannot execute two actions at the same time.
:- occurs(A1,I), occurs(A2,I), A1!=A2.

something_happened(I) :- occurs(A, I).
:- not goal(I),
   not something_happened(I).




%%%%%%%%%
%% Goal:
%%%%%%%%%
goal(I) :- goal_height(L), holds(loc(B, L), I).



%%%%%%%%%%%%%%%%%%%%%%%
%% Initial Conditions:
%%%%%%%%%%%%%%%%%%%%%%%
holds(loc(blue_box, blue_place),0).
holds(loc(orange_box, orange_place),0).
holds(loc(yellow_box, yellow_place),0).
holds(loc(red_box, red_place),0).
holds(loc(green_box, green_place),0).
-holds(in_hand(blue_box),0).
-holds(in_hand(orange_box),0).
-holds(in_hand(yellow_box),0).
-holds(in_hand(red_box),0).
-holds(in_hand(green_box),0).
holds(empty(1),0).
holds(empty(2),0).
holds(empty(3),0).
holds(empty(4),0).
holds(empty(5),0).



%%%%%%%%%%%%%%%%%%
%% Learned axioms:
%%%%%%%%%%%%%%%%%%
holds(unstable_placement(red_box),I) :- holds(loc(green_box,1),I).
holds(unstable_placement(yellow_box),I) :- holds(loc(green_box,1),I).
holds(unstable_placement(blue_box),I) :- holds(loc(green_box,1),I).
holds(unstable_placement(orange_box),I) :- holds(loc(green_box,1),I).
holds(unstable_placement(blue_box),I) :- holds(loc(orange_box,1),I).
holds(unstable_placement(blue_box),I) :- holds(loc(red_box,1),I).
holds(unstable_placement(blue_box),I) :- holds(loc(yellow_box,1),I).
holds(unstable_placement(blue_box),I) :- holds(loc(green_box,1),I).



%%%%%%%%%%%%
%% History:
%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
display
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
occurs.
