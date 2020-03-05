import React, { Component } from 'react';
import { Link } from 'react-router-dom';
import { connect } from 'react-redux';
import IntlMessages from '../../components/utility/intlMessages';
import SingleCart from '../../components/cart/singleCartModal';
import Scrollbars from '../../components/utility/customScrollBar';
// import ecommerceAction from "../../redux/ecommerce/actions";
import { SidebarContent } from './sidebarNotification.style';

let totalPrice;
class TopbarAddtoCart extends Component {
  componentDidMount() {
    const { loadingInitData, initData } = this.props;
    if (!loadingInitData) {
      initData();
    }
  }

  renderProducts = () => {
    const { productQuantity, products } = this.props;
    totalPrice = 0;
    if (!productQuantity || productQuantity.length === 0) {
      return <span className="noItemMsg">No item found</span>;
    }
    return productQuantity.map(product => {
      totalPrice += product.quantity * products[product.objectID].price;
      return (
        <SingleCart
          key={product.objectID}
          quantity={product.quantity}
          changeQuantity={this.changeQuantity}
          cancelQuantity={this.cancelQuantity}
          {...products[product.objectID]}
        />
      );
    });
  };
  changeQuantity = (objectID, quantity) => {
    const { productQuantity } = this.props;
    const newProductQuantity = [];
    productQuantity.forEach(product => {
      if (product.objectID !== objectID) {
        newProductQuantity.push(product);
      } else {
        newProductQuantity.push({
          objectID,
          quantity,
        });
      }
    });
    this.props.changeProductQuantity(newProductQuantity);
  };
  cancelQuantity = objectID => {
    const { productQuantity } = this.props;
    const newProductQuantity = [];
    productQuantity.forEach(product => {
      if (product.objectID !== objectID) {
        newProductQuantity.push(product);
      }
    });
    this.props.changeProductQuantity(newProductQuantity);
  };
  render() {
    const { url, height } = this.props;
    return (
      <SidebarContent style={{ height: height - 65 }}>
        <div className="dropdownBody cartItemsWrapper">
          <Scrollbars style={{ height: '100%' }}>
            {this.renderProducts()}
          </Scrollbars>
        </div>
        <div className="dropdownFooterLinks">
          <Link to={`${url}/cart`} onClick={this.hide}>
            <IntlMessages id="topbar.viewCart" />
          </Link>

          <h3>
            <IntlMessages id="topbar.totalPrice" />:{' '}
            <span>${totalPrice.toFixed(2)}</span>
          </h3>
        </div>
      </SidebarContent>
    );
  }
}
function mapStateToProps(state) {
  return {
    auth: state.Auth,
    ...state.App,
    ...state.Ecommerce,
    customizedTheme: state.ThemeSwitcher.topbarTheme,
    height: state.App.height,
  };
}
export default connect(
  mapStateToProps
  // ecommerceAction
)(TopbarAddtoCart);
