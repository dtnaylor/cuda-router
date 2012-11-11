#ifndef CLICK_FILTERBYHOPS_HH
#define CLICK_FILTERBYHOPS_HH
#include <click/element.hh>
CLICK_DECLS

/*
 * =c
 * FilterByGridHops(HOPS)
 * =s Grid
 * =d
 *
 * Expects GRID_NBR_ENCAP packets with MAC headers on input 0.  Any
 * packet that has travelled less than HOPS so fare is sent to
 * output 0.  Packets that have already travelled HOPS or more are
 * sent out output 1.  HOPS is an Integer.
 *
 * =a FilterByRange
 */

class FilterByGridHops : public Element {

public:

  FilterByGridHops();
  ~FilterByGridHops();

  const char *class_name() const		{ return "FilterByGridHops"; }
  const char *port_count() const		{ return "1/2"; }
  const char *processing() const		{ return PUSH; }

  int configure(Vector<String> &, ErrorHandler *);
  int initialize(ErrorHandler *);

  void push(int port, Packet *);
private:
  unsigned int _max_hops;
};

CLICK_ENDDECLS
#endif
