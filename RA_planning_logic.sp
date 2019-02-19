#const numSteps = 10.
#const numMessages = 0.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  sorts
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#step = 0..numSteps.
#id = 0..numMessages.

#location = {bobs_office, johns_office, sarahs_office, sallys_office, kitchen, library, bathroom}.
#person = {bob, john, sarah, sally}.
#robot = {rob1}.
#status = {delivered, undelivered}.
#agent = #robot + #person.
#boolean = {true, false}.

#inertial_fluent = loc(#agent, #location) + message_status(#id, #person, #status).
#fluent = #inertial_fluent.

#action = move(#robot, #location) + deliver(#robot, #id, #person).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
predicates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
workplace(#person, #location).
default_negated(#person, #location).
next_to(#location, #location).
holds(#fluent, #step).
occurs(#action, #step).
obs(#fluent, #boolean, #step).
hpd(#action, #step).
success().
goal(#step).
something_happened(#step).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 rules
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% State Constraints %%

%% A person's default location is their workplace
holds(loc(P, L), 0) :- not default_negated(P, L), workplace(P, L).
default_negated(P, L) :- obs(loc(P, L2), true, I), L!=L2.
default_negated(P, L) :- obs(loc(P, L), false, I).

%% next_to is transitive
next_to(L1, L2) :- next_to(L2, L1).

%% An agent cannot be in more than one location
-holds(loc(A, L1), I) :- holds(loc(A, L2), I), L1!=L2.

%% A message cannot be both delivered and undelivered
:- holds(message_status(ID, P, delivered), I), holds(message_status(ID, P, undelivered), I).



%% Causal Laws %%

holds(message_status(ID, P, delivered), I+1) :- occurs(deliver(R, ID, P), I).
-holds(message_status(ID, P, undelivered), I+1) :- occurs(deliver(R, ID, P), I).

holds(loc(R, L), I+1) :- occurs(move(R, L), I).



%% Executability Conditions %%

%% Cannot move to a location you are already in
-occurs(move(R, L), I) :- holds(loc(R, L), I).

%% Cannot deliver a message if its intended recipient is not in the room
-occurs(deliver(R, ID, P), I) :- holds(loc(R, L), I), not holds(loc(P, L), I).

%% Cannot deliver a message if it has already been delivered
-occurs(deliver(R, ID, P), I) :- holds(message_status(ID, P, delivered), I).

%% Cannot move to a location that is not next to the current location
-occurs(move(R, L1), I) :- holds(loc(R, L2), I), not next_to(L1, L2).



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Inertial axiom + CWA

%% General inertia axioms.
holds(F,I+1) :- #inertial_fluent(F), holds(F,I), not -holds(F,I+1).

-holds(F,I+1) :- #inertial_fluent(F), -holds(F,I), not holds(F,I+1).

%% CWA for Actions.
-occurs(A,I) :- not occurs(A,I).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Take what actually happened into account.
occurs(A,I) :- hpd(A,I).


%% Reality check axioms.
:- obs(F, true, I), -holds(F, I).
:- obs(F, false, I), holds(F, I).

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



%%%%%%%%%%%%%%%%%%%%
%% Location layout:
%%%%%%%%%%%%%%%%%%%%
next_to(kitchen, bathroom).
next_to(kitchen, library).
next_to(bobs_office, library).
next_to(johns_office, library).
next_to(sarahs_office, kitchen).
next_to(sallys_office, kitchen).



%%%%%%%%%
%% Goal:
%%%%%%%%%
goal(I) :- holds(message_status(0, sarah, delivered), I), holds(loc(rob1, L), I), holds(loc(sally, L), I).



%%%%%%%%%%%%%%%%%%%%%%%
%% Initial Conditions:
%%%%%%%%%%%%%%%%%%%%%%%
holds(loc(rob1, library), 0).
holds(message_status(0, sarah, undelivered), 0).
obs(loc(john, library), false, 0).
obs(loc(bob, library), false, 0).
obs(loc(sarah, library), false, 0).
obs(loc(sally, library), true, 0).



%%%%%%%%%%%%%%%%%%%
%% Learned axioms:
%%%%%%%%%%%%%%%%%%%
workplace(bob, bobs_office).
workplace(sarah, sarahs_office).
workplace(john, johns_office).
workplace(sally, sallys_office).



%%%%%%%%%%%%
%% History:
%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
display
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
occurs.
