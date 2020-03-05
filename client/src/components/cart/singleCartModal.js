import React, { Component } from 'react';
import { notification } from '../index';
import TopbarCartWrapper, { Icon } from './topbarCartDropdown.style';

export default class extends Component {
  onChange = value => {
    if (!isNaN(value)) {
      if (value !== this.props.quantity) {
        this.props.changeQuantity(this.props.objectID, value);
      }
    } else {
      notification('error', 'Please give valid number');
    }
  };

  render() {
    const {
      price,
      quantity,
      _highlightResult,
      image,
      objectID,
      cancelQuantity,
    } = this.props;
    return (
      <TopbarCartWrapper className="cartItems">
        <div className="itemImage">
          <img alt="#" src={image} />
        </div>
        <div className="cartDetails">
          <h3>
            <a href="#!">{_highlightResult.name.value}</a>
          </h3>
          <p className="itemPriceQuantity">
            <span>$</span>
            <span>{price.toFixed(2)}</span>
            <span className="itemMultiplier">X</span>
            <span className="itemQuantity">{quantity}</span>
          </p>
        </div>
        <a
          href="#!"
          className="itemRemove"
          onClick={() => cancelQuantity(objectID)}
        >
          <Icon>clear</Icon>
        </a>
      </TopbarCartWrapper>
    );
  }
}
