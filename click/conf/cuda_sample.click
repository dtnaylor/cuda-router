elementclass PacketSink {
    |
    input -> Socket(UDP, 127.0.0.1, 9877, CLIENT true);
    Socket(UDP, 127.0.0.1, 9878) -> MarkIPHeader() -> Queue -> output;
}

RandomSource(64) -> UDPIPEncap(123.123.123.123, 1234, 210.210.210.210, 4321) -> //IPPrint("out of source") ->
PacketSink -> //IPPrint("out of sink") ->
Discard();