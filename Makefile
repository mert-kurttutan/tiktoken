.PHONY: test

DATA_DIR = data

dir_guard=@mkdir -p $(@D)

TESTS_RESOURCES = $(DATA_DIR)/big.txt

# Launch the test suite
test: $(TESTS_RESOURCES)
	pytest

$(DATA_DIR)/big.txt :
	$(dir_guard)
	wget https://norvig.com/big.txt -O $@

$(DATA_DIR)/small.txt : $(DATA_DIR)/big.txt
	head -100 $(DATA_DIR)/big.txt > $@
