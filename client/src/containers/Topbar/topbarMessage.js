import React, { Component } from 'react';
import { findDOMNode } from 'react-dom';
import { connect } from 'react-redux';
import IntlMessages from '../../components/utility/intlMessages';
import Scrollbars from '../../components/utility/customScrollBar';
import { SidebarContent } from './sidebarNotification.style';
import Image from '../../images/user.jpg';
import Image1 from '../../images/user-1.jpg';

const demoMassage = [
  {
    id: 1,
    img: Image,
    name: 'David Doe',
    time: '3 minutes ago',
    massage:
      'A National Book Award Finalist An Edgar Award Finalist A California Book Award Gold Medal Winner',
  },
  {
    id: 2,
    img: Image1,
    name: 'Navis Doe',
    time: '4 minutes ago',
    massage:
      'A National Book Award Finalist An Edgar Award Finalist A California Book Award Gold Medal Winner',
  },
  {
    id: 3,
    img: Image,
    name: 'Emanual Doe',
    time: '5 minutes ago',
    massage:
      'A National Book Award Finalist An Edgar Award Finalist A California Book Award Gold Medal Winner',
  },
  {
    id: 4,
    img: Image1,
    name: 'Dowain Doe',
    time: '6 minutes ago',
    massage:
      'A National Book Award Finalist An Edgar Award Finalist A California Book Award Gold Medal Winner',
  },
  {
    id: 5,
    img: Image,
    name: 'James Doe',
    time: '6 minutes ago',
    massage:
      'A National Book Award Finalist An Edgar Award Finalist A California Book Award Gold Medal Winner',
  },
  {
    id: 6,
    img: Image1,
    name: 'Ralph Doe',
    time: '6 minutes ago',
    massage:
      'A National Book Award Finalist An Edgar Award Finalist A California Book Award Gold Medal Winner',
  },
  {
    id: 7,
    img: Image,
    name: 'Brook Doe',
    time: '6 minutes ago',
    massage:
      'A National Book Award Finalist An Edgar Award Finalist A California Book Award Gold Medal Winner',
  },
  {
    id: 8,
    img: Image1,
    name: 'Raphayel Doe',
    time: '6 minutes ago',
    massage:
      'A National Book Award Finalist An Edgar Award Finalist A California Book Award Gold Medal Winner',
  },
];
class TopbarMessage extends Component {
  state = {
    visible: false,
    anchorEl: null,
  };
  hide = () => {
    this.setState({ visible: false });
  };
  handleVisibleChange = () => {
    this.setState({
      visible: !this.state.visible,
      anchorEl: findDOMNode(this.button),
    });
  };
  render() {
    const { height } = this.props;
    return (
      <SidebarContent
        className="topbarMessage withImg"
        style={{ height: height - 65 }}
      >
        <div className="dropdownBody">
          <Scrollbars style={{ height: '100%' }}>
            {demoMassage.map(massage => (
              <a href="#!" className="dropdownListItem" key={massage.id}>
                <div className="userImgWrapper">
                  <img alt="#" src={massage.img} />
                </div>

                <div className="listContent">
                  <div className="listHead">
                    <h5>{massage.name}</h5>
                    <span className="date">{massage.time}</span>
                  </div>
                  <p>{massage.massage}</p>
                </div>
              </a>
            ))}
          </Scrollbars>
        </div>
        <a href="#!" className="viewAllBtn">
          <IntlMessages id="topbar.viewAll" />
        </a>
      </SidebarContent>
    );
  }
}

export default connect(state => ({
  ...state.App,
  customizedTheme: state.ThemeSwitcher.topbarTheme,
  height: state.App.height,
}))(TopbarMessage);
