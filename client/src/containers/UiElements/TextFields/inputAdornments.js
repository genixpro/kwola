import React from "react";
import PropTypes from "prop-types";
import { withStyles } from '@material-ui/core/styles';
import { IconButton } from "../../../components/uielements/button";
import Input from "../../../components/uielements/input";
import {
  InputLabel,
  InputAdornment,
  Container,
  FormControl,
  FormHelperText
} from "./textfield.style";
import Icon from "../../../components/uielements/icon/index.js";

class InputAdornments extends React.Component {
  state = {
    amount: "",
    password: "",
    weight: "",
    showPassword: false
  };

  handleChange = prop => event => {
    this.setState({ [prop]: event.target.value });
  };

  handleMouseDownPassword = event => {
    event.preventDefault();
  };

  handleClickShowPasssword = () => {
    this.setState({ showPassword: !this.state.showPassword });
  };

  render() {
    return (
      <Container>
        <FormControl fullWidth>
          <InputLabel htmlFor="amount">Amount</InputLabel>
          <Input
            id="amount"
            value={this.state.amount}
            onChange={this.handleChange("amount")}
            startAdornment={<InputAdornment position="start">$</InputAdornment>}
          />
        </FormControl>
        <FormControl className="withoutLabel">
          <Input
            id="weight"
            value={this.state.weight}
            onChange={this.handleChange("weight")}
            endAdornment={<InputAdornment position="end">Kg</InputAdornment>}
          />
          <FormHelperText>Weight</FormHelperText>
        </FormControl>
        <FormControl>
          <InputLabel htmlFor="password">Password</InputLabel>
          <Input
            id="password"
            type={this.state.showPassword ? "text" : "password"}
            value={this.state.password}
            onChange={this.handleChange("password")}
            endAdornment={
              <InputAdornment position="end">
                <IconButton
                  onClick={this.handleClickShowPasssword}
                  onMouseDown={this.handleMouseDownPassword}
                >
                  {this.state.showPassword ? (
                    <Icon>visibility_off</Icon>
                  ) : (
                    <Icon>visibility</Icon>
                  )}
                </IconButton>
              </InputAdornment>
            }
          />
        </FormControl>
      </Container>
    );
  }
}

InputAdornments.propTypes = {
  classes: PropTypes.object.isRequired
};

export default withStyles({})(InputAdornments);
