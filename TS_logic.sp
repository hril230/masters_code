%%%%%%%%%%%%%%%%%%%%%%%%%
sorts
%%%%%%%%%%%%%%%%%%%%%%%%%
#shape = {triangle, inverted_triangle, circle, hexagon, tall_rectangle, wide_rectangle, square, diamond}.
#main_colour = {white, red, blue}.
#border_colour = {white, red, blue}.
#background_image = {none, triangle, square}.
#symbol = {blank, bumpy_road, bump, slippery_road, left_turn, right_turn, left_then_right, right_then_left, people, cyclist, cow, road_worker, traffic_lights, fence, danger, narrow_road, narrow_on_left, narrow_on_right, priority_to_through_traffic, cross, oncoming_traffic, stop, do_not_enter, weight_limit, truck, width_limit, height_limit, no_passing, speed_limit, arrow, side_arrow, straight_or_right, yield, paths, numbers, p, children, barrier, pedestrian}.
#secondary_symbol = {none, people, one_and_fifteen, sixteen_and_thirtyone, disabled, car, truck, bus, fence, arrow}.
#cross = {none, red, black}.
#class = 0..61.
#rule = 0..90.
#question = 0..3.
#answer = 0..156.


%%%%%%%%%%%%%%%%%%%%%%%%
predicates
%%%%%%%%%%%%%%%%%%%%%%%%
shape(#shape).
main_colour(#main_colour).
border_colour(#border_colour).
background_image(#background_image).
symbol(#symbol).
secondary_symbol(#secondary_symbol).
cross(#cross).
class(#class).
rule(#rule).
question(#question).
answer(#answer).
question_class(#class).


%%%%%%%%%%%%%%%%%%%%%%%%
rules
%%%%%%%%%%%%%%%%%%%%%%%%

-class(C1) :- class(C2), C1!=C2.

%% Logic for classifying
rule(0) :- symbol(bumpy_road).
class(0) :- rule(0).
rule(1) :- shape(triangle), symbol(bump).
rule(2) :- main_colour(white), symbol(bump).
rule(3) :- border_colour(red), symbol(bump).
rule(4) :- background_image(none), symbol(bump).
class(1) :- rule(1).
class(1) :- rule(2).
class(1) :- rule(3).
class(1) :- rule(4).
rule(5) :- symbol(slippery_road).
class(2) :- rule(5).
rule(6) :- shape(triangle), symbol(left_turn).
rule(7) :- cross(none), symbol(left_turn).
class(3) :- rule(6).
class(3) :- rule(7).
rule(8) :- shape(triangle), symbol(right_turn).
rule(9) :- cross(none), symbol(right_turn).
class(4) :- rule(8).
class(4) :- rule(9).
rule(10) :- symbol(left_then_right).
class(5) :- rule(10).
rule(11) :- symbol(right_then_left).
class(6) :- rule(11).
rule(12) :- symbol(people).
class(7) :- rule(12).
rule(13) :- shape(triangle), symbol(cyclist).
class(8) :- rule(13).
rule(14) :- symbol(cow).
class(9) :- rule(14).
rule(15) :- shape(triangle), symbol(road_worker).
rule(16) :- main_colour(white), symbol(road_worker).
rule(17) :- border_colour(red), symbol(road_worker).
class(10) :- rule(15).
class(10) :- rule(16).
class(10) :- rule(17).
rule(18) :- symbol(traffic_lights).
class(11) :- rule(18).
rule(19) :- symbol(fence).
class(12) :- rule(19).
rule(20) :- symbol(danger).
class(13) :- rule(20).
rule(21) :- symbol(narrow_road).
class(14) :- rule(21).
rule(22) :- symbol(narrow_on_left).
class(15) :- rule(22).
rule(23) :- symbol(narrow_on_right).
class(16) :- rule(23).
rule(24) :- symbol(priority_to_through_traffic).
class(17) :- rule(24).
rule(25) :- shape(triangle), symbol(cross).
rule(26) :- main_colour(white), symbol(cross).
class(18) :- rule(25).
class(18) :- rule(26).
rule(27) :- shape(inverted_triangle).
class(19) :- rule(27).
rule(28) :- shape(circle), symbol(oncoming_traffic).
rule(29) :- main_colour(white), symbol(oncoming_traffic).
rule(30) :- border_colour(red), symbol(oncoming_traffic).
class(20) :- rule(28).
class(20) :- rule(29).
class(20) :- rule(30).
rule(31) :- shape(hexagon).
rule(32) :- symbol(stop).
class(21) :- rule(31).
class(21) :- rule(32).
rule(33) :- main_colour(red), border_colour(red).
rule(34) :- main_colour(red), shape(circle).
rule(35) :- symbol(do_not_enter).
class(22) :- rule(33).
class(22) :- rule(34).
class(22) :- rule(35).
rule(36) :- border_colour(red), shape(circle), symbol(cyclist).
class(23) :- rule(36).
rule(37) :- symbol(weight_limit).
class(24) :- rule(37).
rule(38) :- symbol(truck).
class(25) :- rule(38).
rule(39) :- symbol(width_limit).
class(26) :- rule(39).
rule(40) :- symbol(height_limit).
class(27) :- rule(40).
rule(41) :- shape(circle), main_colour(white), symbol(blank).
class(28) :- rule(41).
rule(42) :- cross(red), symbol(left_turn).
rule(43) :- shape(circle), symbol(left_turn).
class(29) :- rule(42).
class(29) :- rule(43).
rule(44) :- cross(red), symbol(right_turn).
rule(45) :- shape(circle), symbol(right_turn).
class(30) :- rule(44).
class(30) :- rule(45).
rule(46) :- symbol(no_passing).
class(31) :- rule(46).
rule(47) :- symbol(speed_limit).
class(32) :- rule(47).
rule(48) :- secondary_symbol(people).
class(33) :- rule(48).
rule(49) :- shape(circle), symbol(arrow).
class(34) :- rule(49).
rule(50) :- symbol(side_arrow).
class(35) :- rule(50).
rule(51) :- symbol(straight_or_right).
class(36) :- rule(51).
rule(52) :- symbol(yield).
class(37) :- rule(52).
rule(53):- main_colour(blue), symbol(cyclist), background_image(none), secondary_symbol(none).
rule(54):- border_colour(white), symbol(cyclist), background_image(none), secondary_symbol(none).
class(38) :- rule(53).
class(38) :- rule(54).
rule(55) :- symbol(paths).
class(39) :- rule(55).
rule(56) :- main_colour(blue), border_colour(red), secondary_symbol(none), cross(red).
rule(57) :- main_colour(blue), symbol(blank).
rule(58) :- cross(red), symbol(blank).
class(40) :- rule(56).
class(40) :- rule(57).
class(40) :- rule(58).
rule(59) :- main_colour(blue), border_colour(red), secondary_symbol(none), cross(none).
rule(60) :- shape(circle), symbol(cross).
rule(61) :- main_colour(blue), symbol(cross).
class(41) :- rule(59).
class(41) :- rule(60).
class(41) :- rule(61).
rule(62) :- secondary_symbol(one_and_fifteen).
class(42) :- rule(62).
rule(63) :- secondary_symbol(sixteen_and_thirtyone).
class(43) :- rule(63).
rule(64) :- shape(tall_rectangle), symbol(oncoming_traffic).
rule(65) :- main_colour(blue), symbol(oncoming_traffic).
rule(66) :- border_colour(white), symbol(oncoming_traffic).
class(44) :- rule(64).
class(44) :- rule(65).
class(44) :- rule(66).
rule(67) :- border_colour(blue), secondary_symbol(none).
class(45) :- rule(67).
rule(68) :- secondary_symbol(disabled).
class(46) :- rule(68).
rule(69) :- secondary_symbol(car).
class(47) :- rule(69).
rule(70) :- secondary_symbol(truck).
class(48) :- rule(70).
rule(71) :- secondary_symbol(bus).
class(49) :- rule(71).
rule(72) :- secondary_symbol(fence).
class(50) :- rule(72).
rule(73) :- shape(wide_rectangle), cross(none).
rule(74) :- symbol(children), cross(none).
class(51) :- rule(73).
class(51) :- rule(74).
rule(75) :- shape(wide_rectangle), cross(red).
rule(76) :- symbol(children), cross(red).
class(52) :- rule(75).
class(52) :- rule(76).
rule(77) :- shape(square), background_image(none).
rule(78) :- shape(square), symbol(arrow).
class(53) :- rule(77).
class(53) :- rule(78).
rule(79) :- symbol(barrier).
class(54) :- rule(79).
rule(80) :- symbol(road_worker), cross(red).
class(55) :- rule(80).
rule(81) :- symbol(pedestrian).
class(56) :- rule(81).
rule(82) :- background_image(triangle), symbol(cyclist).
rule(83) :- shape(square), symbol(cyclist).
class(57) :- rule(82).
class(57) :- rule(83).
rule(84) :- secondary_symbol(arrow).
class(58) :- rule(84).
rule(85) :- symbol(bump), shape(square).
rule(86) :- symbol(bump), main_colour(blue).
rule(87) :- symbol(bump), border_colour(white).
rule(88) :- symbol(bump), background_image(triangle).
class(59) :- rule(85).
class(59) :- rule(86).
class(59) :- rule(87).
class(59) :- rule(88).
rule(89) :- cross(black).
class(60) :- rule(89).
rule(90) :- shape(diamond), cross(none).
class(61) :- rule(90).

%% Logic for question answering
% answers to question zero (what type of sign is this?)
answer(0) :- class(0), question(0).
answer(1) :- class(1), question(0).
answer(2) :- class(2), question(0).
answer(3) :- class(3), question(0).
answer(4) :- class(4), question(0).
answer(5) :- class(5), question(0).
answer(6) :- class(6), question(0).
answer(7) :- class(7), question(0).
answer(8) :- class(8), question(0).
answer(9) :- class(9), question(0).
answer(10) :- class(10), question(0).
answer(11) :- class(11), question(0).
answer(12) :- class(12), question(0).
answer(13) :- class(13), question(0).
answer(14) :- class(14), question(0).
answer(15) :- class(15), question(0).
answer(16) :- class(16), question(0).
answer(17) :- class(17), question(0).
answer(18) :- class(18), question(0).
answer(19) :- class(19), question(0).
answer(20) :- class(20), question(0).
answer(21) :- class(21), question(0).
answer(22) :- class(22), question(0).
answer(23) :- class(23), question(0).
answer(24) :- class(24), question(0).
answer(25) :- class(25), question(0).
answer(26) :- class(26), question(0).
answer(27) :- class(27), question(0).
answer(28) :- class(28), question(0).
answer(29) :- class(29), question(0).
answer(30) :- class(30), question(0).
answer(31) :- class(31), question(0).
answer(32) :- class(32), question(0).
answer(33) :- class(33), question(0).
answer(34) :- class(34), question(0).
answer(35) :- class(35), question(0).
answer(36) :- class(36), question(0).
answer(37) :- class(37), question(0).
answer(38) :- class(38), question(0).
answer(39) :- class(39), question(0).
answer(40) :- class(40), question(0).
answer(41) :- class(41), question(0).
answer(42) :- class(42), question(0).
answer(42) :- class(43), question(0).
answer(43) :- class(44), question(0).
answer(44) :- class(45), question(0).
answer(45) :- class(46), question(0).
answer(46) :- class(47), question(0).
answer(47) :- class(48), question(0).
answer(48) :- class(49), question(0).
answer(44) :- class(50), question(0).
answer(49) :- class(51), question(0).
answer(50) :- class(52), question(0).
answer(34) :- class(53), question(0).
answer(51) :- class(54), question(0).
answer(52) :- class(55), question(0).
answer(53) :- class(56), question(0).
answer(54) :- class(57), question(0).
answer(55) :- class(58), question(0).
answer(56) :- class(59), question(0).
answer(57) :- class(60), question(0).
answer(58) :- class(61), question(0).
% answers to question one (what is the sign's message?)
answer(59) :- class(0), question(1).
answer(60) :- class(1), question(1).
answer(61) :- class(2), question(1).
answer(62) :- class(3), question(1).
answer(63) :- class(4), question(1).
answer(64) :- class(5), question(1).
answer(65) :- class(6), question(1).
answer(66) :- class(7), question(1).
answer(67) :- class(8), question(1).
answer(68) :- class(9), question(1).
answer(69) :- class(10), question(1).
answer(70) :- class(11), question(1).
answer(71) :- class(12), question(1).
answer(72) :- class(13), question(1).
answer(73) :- class(14), question(1).
answer(74) :- class(15), question(1).
answer(75) :- class(16), question(1).
answer(76) :- class(17), question(1).
answer(77) :- class(18), question(1).
answer(78) :- class(19), question(1).
answer(79) :- class(20), question(1).
answer(80) :- class(21), question(1).
answer(22) :- class(22), question(1).
answer(81) :- class(23), question(1).
answer(82) :- class(24), question(1).
answer(83) :- class(25), question(1).
answer(84) :- class(26), question(1).
answer(85) :- class(27), question(1).
answer(86) :- class(28), question(1).
answer(87) :- class(29), question(1).
answer(88) :- class(30), question(1).
answer(89) :- class(31), question(1).
answer(90) :- class(32), question(1).
answer(91) :- class(33), question(1).
answer(92) :- class(34), question(1).
answer(93) :- class(35), question(1).
answer(94) :- class(36), question(1).
answer(95) :- class(37), question(1).
answer(96) :- class(38), question(1).
answer(97) :- class(39), question(1).
answer(98) :- class(40), question(1).
answer(99) :- class(41), question(1).
answer(42) :- class(42), question(1).
answer(42) :- class(43), question(1).
answer(100) :- class(44), question(1).
answer(101) :- class(45), question(1).
answer(102) :- class(46), question(1).
answer(103) :- class(47), question(1).
answer(104) :- class(48), question(1).
answer(105) :- class(49), question(1).
answer(101) :- class(50), question(1).
answer(106) :- class(51), question(1).
answer(107) :- class(52), question(1).
answer(92) :- class(53), question(1).
answer(108) :- class(54), question(1).
answer(109) :- class(55), question(1).
answer(110) :- class(56), question(1).
answer(111) :- class(57), question(1).
answer(112) :- class(58), question(1).
answer(113) :- class(59), question(1).
answer(57) :- class(60), question(1).
answer(58) :- class(61), question(1).
% answers to question two (how should the driver react to this sign?)
answer(114) :- class(0), question(2).
answer(115) :- class(1), question(2).
answer(114) :- class(2), question(2).
answer(116) :- class(3), question(2).
answer(117) :- class(4), question(2).
answer(116) :- class(5), question(2).
answer(117) :- class(6), question(2).
answer(114) :- class(7), question(2).
answer(114) :- class(8), question(2).
answer(114) :- class(9), question(2).
answer(114) :- class(10), question(2).
answer(118) :- class(11), question(2).
answer(119) :- class(12), question(2).
answer(114) :- class(13), question(2).
answer(114) :- class(14), question(2).
answer(116) :- class(15), question(2).
answer(117) :- class(16), question(2).
answer(120) :- class(17), question(2).
answer(114) :- class(18), question(2).
answer(121) :- class(19), question(2).
answer(122) :- class(20), question(2).
answer(123) :- class(21), question(2).
answer(124) :- class(22), question(2).
answer(125) :- class(23), question(2).
answer(126) :- class(24), question(2).
answer(127) :- class(25), question(2).
answer(128) :- class(26), question(2).
answer(129) :- class(27), question(2).
answer(130) :- class(28), question(2).
answer(94) :- class(29), question(2).
answer(131) :- class(30), question(2).
answer(132) :- class(31), question(2).
answer(133) :- class(32), question(2).
answer(134) :- class(33), question(2).
answer(135) :- class(34), question(2).
answer(136) :- class(35), question(2).
answer(94) :- class(36), question(2).
answer(137) :- class(37), question(2).
answer(138) :- class(38), question(2).
answer(139) :- class(39), question(2).
answer(140) :- class(40), question(2).
answer(141) :- class(41), question(2).
answer(42) :- class(42), question(2).
answer(42) :- class(43), question(2).
answer(142) :- class(44), question(2).
answer(143) :- class(45), question(2).
answer(144) :- class(46), question(2).
answer(145) :- class(47), question(2).
answer(146) :- class(48), question(2).
answer(147) :- class(49), question(2).
answer(143) :- class(50), question(2).
answer(114) :- class(51), question(2).
answer(148) :- class(52), question(2).
answer(135) :- class(53), question(2).
answer(149) :- class(54), question(2).
answer(148) :- class(55), question(2).
answer(150) :- class(56), question(2).
answer(151) :- class(57), question(2).
answer(152) :- class(58), question(2).
answer(153) :- class(59), question(2).
answer(154) :- class(60), question(2).
answer(148) :- class(61), question(2).
% answers to question three (is this a [insert sign type here] sign?)
answer(155) :- question(3), question_class(C), class(C).
answer(156) :- question(3), question_class(C1), class(C2), C1!=C2.
