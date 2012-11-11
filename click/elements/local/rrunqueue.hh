#ifndef CLICK_RRUNQUEUE_HH
#define CLICK_RRUNQUEUE_HH
#include <click/element.hh>
#include <click/task.hh>
CLICK_DECLS

/*
 * =c
 * RoundRobinUnqueue([BURSTSIZE])
 * =s scheduling
 * pull-to-push converter
 * =d
 * Pulls packets from input ports in a round robin fashion, then pushes them
 * out the output corresponding to the input that the packet came from. Pulls
 * a maximum of BURSTSIZE packets every time it is scheduled. Default
 * BURSTSIZE is 1. If BURSTSIZE is 0, pull until nothing comes back.
 *
 * =a Unqueue, RatedUnqueue, BandwidthRatedUnqueue
 */

class RoundRobinUnqueue : public Element { public:

  RoundRobinUnqueue();
  ~RoundRobinUnqueue();

  const char *class_name() const	{ return "RoundRobinUnqueue"; }
  const char *port_count() const	{ return "-/-"; }
  const char *processing() const	{ return PULL_TO_PUSH; }

  int configure(Vector<String> &, ErrorHandler *);
  int initialize(ErrorHandler *);
  void add_handlers();

  bool run_task(Task *);

  static String read_param(Element *e, void *);

 private:

  int _burst;
  unsigned _packets;
  Task _task;
  int _next;
};

CLICK_ENDDECLS
#endif
